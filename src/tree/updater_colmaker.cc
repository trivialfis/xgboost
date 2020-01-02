/*!
 * Copyright 2014-2019 by Contributors
 * \file updater_colmaker.cc
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <bits/stdint-intn.h>
#include <rabit/rabit.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/parameter.h"
#include "xgboost/tree_updater.h"
#include "xgboost/logging.h"
#include "xgboost/json.h"
#include "param.h"
#include "constraints.h"
#include "../common/random.h"
#include "../common/bitmap.h"
#include "split_evaluator.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_colmaker);

enum DefaultDir {
  kLearn = 0,
  kLeft = 1,
  kRight = 2
};

struct ColMakerTrainParam : XGBoostParameter<ColMakerTrainParam> {
  // speed optimization for dense column
  float opt_dense_col;
  DMLC_DECLARE_PARAMETER(ColMakerTrainParam) {
    DMLC_DECLARE_FIELD(opt_dense_col)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("EXP Param: speed optimization for dense column.");
  }

  /*! \brief whether need forward small to big search: default right */
  inline bool NeedForwardSearch(int default_direction, float col_density,
                                bool indicator) const {
    return default_direction == kRight ||
           (default_direction == kLearn && (col_density < opt_dense_col) &&
            !indicator);
  }
  /*! \brief whether need backward big to small search: default left */
  inline bool NeedBackwardSearch(int default_direction) const {
    return default_direction != kRight;
  }
};

DMLC_REGISTER_PARAMETER(ColMakerTrainParam);

/*! \brief column-wise update to construct a tree */
class ColMaker: public TreeUpdater {
  enum StepDir {
    kForeward,
    kBackward
  };

 public:
  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    colmaker_param_.UpdateAllowUnknown(args);
    if (!spliteval_) {
      spliteval_.reset(SplitEvaluator::Create(param_.split_evaluator));
    }
    spliteval_->Init(&param_);
    if (param_.grow_policy == TrainParam::kLossGuide) {
      LOG(FATAL) << "Exact tree method doesn't support loss guided grow policy.";
    }
  }

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    fromJson(config.at("train_param"), &this->param_);
    fromJson(config.at("colmaker_train_param"), &this->colmaker_param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = toJson(param_);
    out["colmaker_train_param"] = toJson(colmaker_param_);
  }

  char const* Name() const override {
    return "grow_colmaker";
  }

  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    // rescale learning rate according to size of trees
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    interaction_constraints_.Configure(param_, dmat->Info().num_row_);
    // build tree
    for (auto tree : trees) {
      Builder builder(
        param_,
        colmaker_param_,
        std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()),
        this->tparam_->nthread,
        interaction_constraints_);
      builder.Update(gpair->ConstHostVector(), dmat, tree);
    }
    param_.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param_;
  ColMakerTrainParam colmaker_param_;
  // SplitEvaluator that will be cloned for each Builder
  std::unique_ptr<SplitEvaluator> spliteval_;

  FeatureInteractionConstraintHost interaction_constraints_;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    GradStats stats;
    /*! \brief last feature value scanned */
    bst_float last_fvalue;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    ThreadEntry() : last_fvalue{0} {}
  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
    /*! \brief weight calculated related to current data */
    bst_float weight;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    NodeEntry() : root_gain{0.0f}, weight{0.0f} {}
  };
  // actual builder that runs the algorithm
  class Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param,
                     const ColMakerTrainParam& colmaker_train_param,
                     std::unique_ptr<SplitEvaluator> spliteval,
                     int32_t n_threads,
                     FeatureInteractionConstraintHost _interaction_constraints)
        : param_(param), colmaker_train_param_{colmaker_train_param},
          nthread_(n_threads),
          spliteval_(std::move(spliteval)),
          interaction_constraints_{std::move(_interaction_constraints)} {}
    // update one tree, growing
    virtual void Update(const std::vector<GradientPair>& gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree) {
      std::vector<int> newnodes;
      this->InitData(gpair, p_fmat);
      // qexpand_ is just root.
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
      // depth wise build
      for (int depth = 0; depth < param_.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, qexpand_, &newnodes);
        this->InitNewNode(newnodes, gpair, *p_fmat, *p_tree);
        // Apply splits
        for (auto nid : qexpand_) {
          if ((*p_tree)[nid].IsLeaf()) {
            continue;
          }
          int cleft = (*p_tree)[nid].LeftChild();
          int cright = (*p_tree)[nid].RightChild();
          spliteval_->AddSplit(nid, cleft, cright,
                               nodes_statistics_[nid].best.SplitIndex(),
                               nodes_statistics_[cleft].weight,
                               nodes_statistics_[cright].weight);
          interaction_constraints_.Split(
              nid, nodes_statistics_[nid].best.SplitIndex(), cleft, cright);
        }
        qexpand_ = newnodes;
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf, and no-longer "fresh"
      for (const int nid : qexpand_) {
        (*p_tree)[nid].SetLeaf(nodes_statistics_[nid].weight *
                               param_.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->Stat(nid).loss_chg = nodes_statistics_[nid].best.loss_chg;
        p_tree->Stat(nid).base_weight = nodes_statistics_[nid].weight;
        p_tree->Stat(nid).sum_hess =
            static_cast<float>(nodes_statistics_[nid].stats.sum_hess);
      }
    }

   protected:
    // initialize temp data structure
    void InitData(const std::vector<GradientPair>& gpair, DMatrix* fmat) {
      {
        // setup position
        position_.resize(gpair.size());
        std::fill(position_.begin(), position_.end(), 0);
        // mark delete for the deleted datas
        for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
          if (gpair[ridx].GetHess() < 0.0f) {
            position_[ridx] = ~position_[ridx];
          }
        }

        // mark subsample
        if (param_.subsample < 1.0f) {
          std::bernoulli_distribution coin_flip(param_.subsample);
          auto& rnd = common::GlobalRandom();
          for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
            if (gpair[ridx].GetHess() < 0.0f) continue;
            if (!coin_flip(rnd)) {
              position_[ridx] = ~position_[ridx];
            }
          }
        }
      }

      column_sampler_.Init(fmat->Info().num_col_, param_.colsample_bynode,
                           param_.colsample_bylevel, param_.colsample_bytree);

      {
        // setup temp space for each thread
        // reserve a small space
        stemp_.clear();
        stemp_.resize(this->nthread_, std::vector<ThreadEntry>());
        for (auto& i : stemp_) {
          i.clear();
          i.reserve(256);
        }
        nodes_statistics_.reserve(256);
      }

      {
        // expand query
        qexpand_.reserve(256); qexpand_.clear();
        qexpand_.push_back(0);
      }
    }
    /*!
     * \brief initialize the base_weight, root_gain,
     *  and NodeEntry for all the new nodes in qexpand
     */
    inline void InitNewNode(const std::vector<int>& qexpand,
                            const std::vector<GradientPair>& gpair,
                            const DMatrix& fmat,
                            const RegTree& tree) {
      {
        // setup statistics space for each tree node
        for (auto& i : stemp_) {
          i.resize(tree.param.num_nodes);
        }
        nodes_statistics_.resize(tree.param.num_nodes, NodeEntry());
      }

      const MetaInfo& info = fmat.Info();
      // setup position
      const auto ndata = static_cast<bst_omp_uint>(info.num_row_);
#pragma omp parallel for schedule(static)
      for (bst_omp_uint ridx = 0; ridx < ndata; ++ridx) {
        const int tid = omp_get_thread_num();
        if (position_[ridx] < 0) {
          continue;
        }
        // Sum all stat of each node.
        stemp_[tid][position_[ridx]].stats.Add(gpair[ridx]);
      }
      // sum the per thread statistics together
      for (int nid : qexpand) {
        GradStats stats;
        for (auto& s : stemp_) {
          stats.Add(s[nid].stats);
        }
        // update node statistics
        nodes_statistics_[nid].stats = stats;
      }
      // calculating the weights
      for (int nid : qexpand) {
        bst_uint parentid = tree[nid].Parent();
        nodes_statistics_[nid].weight = static_cast<float>(
            spliteval_->ComputeWeight(parentid, nodes_statistics_[nid].stats));
        nodes_statistics_[nid].root_gain = static_cast<float>(
            spliteval_->ComputeScore(parentid, nodes_statistics_[nid].stats,
                                     nodes_statistics_[nid].weight));
      }
    }
    /*! \brief update queue expand add in new leaves */
    inline void UpdateQueueExpand(const RegTree& tree,
                                  const std::vector<int> &qexpand,
                                  std::vector<int>* p_newnodes) {
      p_newnodes->clear();
      for (int nid : qexpand) {
        if (!tree[ nid ].IsLeaf()) {
          p_newnodes->push_back(tree[nid].LeftChild());
          p_newnodes->push_back(tree[nid].RightChild());
        }
      }
    }

    // update enumeration solution
    inline void UpdateEnumeration(int nid, GradientPair gstats,
                                  bst_float fvalue, int d_step, bst_uint fid,
                                  GradStats &c, ThreadEntry* p_node_entry) const { // NOLINT(*)
      // get the statistics of nid
      ThreadEntry& e = *p_node_entry;
      // test if first hit, this is fine, because we set 0 during init
      if (e.stats.Empty()) {
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      } else {
        // try to find a split
        auto SplitIsValid = [](ThreadEntry const& e, float fvalue, float gamma) {
                              return fvalue != e.last_fvalue &&
                                  e.stats.sum_hess >= gamma;
                            };
        if (SplitIsValid(e, fvalue, param_.min_child_weight)) {
          c.SetSubstract(nodes_statistics_[nid].stats, e.stats);
          if (c.sum_hess >= param_.min_child_weight) {
            bst_float loss_chg {0};
            if (d_step == -1) {
              loss_chg = static_cast<bst_float>(
                  spliteval_->ComputeSplitScore(nid, fid, c, e.stats) -
                  nodes_statistics_[nid].root_gain);
              // What is 0.5f?
              bst_float proposed_split = (fvalue + e.last_fvalue) * 0.5f;
              if ( proposed_split == fvalue ) {
                e.best.Update(loss_chg, fid, e.last_fvalue,
                              d_step == -1, c, e.stats);
              } else {
                e.best.Update(loss_chg, fid, proposed_split,
                              d_step == -1, c, e.stats);
              }
            } else {
              loss_chg = static_cast<bst_float>(
                  spliteval_->ComputeSplitScore(nid, fid, e.stats, c) -
                  nodes_statistics_[nid].root_gain);
              bst_float proposed_split = (fvalue + e.last_fvalue) * 0.5f;
              if ( proposed_split == fvalue ) {
                e.best.Update(loss_chg, fid, e.last_fvalue,
                              d_step == -1,
                              /*left_child_sum  = */ e.stats,
                              /*right_child_sum = */ c);
              } else {
                e.best.Update(loss_chg, fid, proposed_split,
                              d_step == -1,
                              e.stats,
                              c);
              }
            }
          }
        }
        // update the statistics
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      }
    }
    // same as EnumerateSplit, with cacheline prefetch optimization
    void EnumerateSplit(const Entry *begin,
                        const Entry *end,
                        int d_step,
                        bst_uint fid,
                        const std::vector<GradientPair> &gpair,
                        std::vector<ThreadEntry> &temp)  const { // NOLINT(*)
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats = GradStats();
      }
      GradStats left_statistics;
      // local cache buffer for position and gradient pair
      constexpr int kBuffer = 32;
      int buf_position[kBuffer] = {};
      GradientPair buf_gpair[kBuffer] = {};
      // aligned ending position
      const Entry *align_end;
      if (d_step > 0) {
        align_end = begin + (end - begin) / kBuffer * kBuffer;
      } else {
        align_end = begin - (begin - end) / kBuffer * kBuffer;
      }
      int i;
      const Entry *it;
      const int align_step = d_step * kBuffer;
      // internal cached loop
      for (it = begin; it != align_end; it += align_step) {
        const Entry *p;
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          buf_position[i] = position_[p->index];
          buf_gpair[i] = gpair[p->index];
        }
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          const int nid = buf_position[i];
          if (nid < 0 || !interaction_constraints_.Query(nid, fid)) { continue; }
          this->UpdateEnumeration(nid, buf_gpair[i], p->fvalue, d_step, fid,
                                  left_statistics, &temp[nid]);
        }
      }

      // finish up the ending piece
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        buf_position[i] = position_[it->index];
        buf_gpair[i] = gpair[it->index];
      }
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        const int nid = buf_position[i];
        if (nid < 0 || !interaction_constraints_.Query(nid, fid)) { continue; }
        this->UpdateEnumeration(nid, buf_gpair[i], it->fvalue, d_step, fid,
                                left_statistics, &temp[nid]);
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {
        ThreadEntry &e = temp[nid];
        left_statistics.SetSubstract(nodes_statistics_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight &&
            left_statistics.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap: -gap;
          if (d_step == -1) {
            loss_chg =
                static_cast<bst_float>(spliteval_->ComputeSplitScore(
                                           nid, fid, left_statistics, e.stats) -
                                       nodes_statistics_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1,
                          left_statistics, e.stats);
          } else {
            loss_chg =
                static_cast<bst_float>(spliteval_->ComputeSplitScore(
                                           nid, fid, e.stats, left_statistics) -
                                       nodes_statistics_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1,
                          e.stats, left_statistics);
          }
        }
      }
    }

    // update the solution candidate
    virtual void UpdateSolution(const SparsePage &batch,
                                const std::vector<bst_feature_t> &feat_set,
                                const std::vector<GradientPair> &gpair,
                                DMatrix* p_fmat) {
      // start enumeration
      const auto num_features = static_cast<bst_omp_uint>(feat_set.size());
#if defined(_OPENMP)
      const int batch_size =  // NOLINT
          std::max(static_cast<int>(num_features / this->nthread_ / 32), 1);
#endif  // defined(_OPENMP)

      {
        std::vector<float> densities(num_features);
        CHECK_EQ(feat_set.size(), num_features);
        for (bst_omp_uint i = 0; i < num_features; ++i) {
          bst_feature_t const fid = feat_set[i];
          densities.at(i) = p_fmat->GetColDensity(fid);
        }

#pragma omp parallel for schedule(dynamic, batch_size)
        for (bst_omp_uint i = 0; i < num_features; ++i) {
          bst_feature_t const fid = feat_set[i];
          int32_t const tid = omp_get_thread_num();
          auto col = batch[fid];
          const bool ind =
              col.size() != 0 && col[0].fvalue == col[col.size() - 1].fvalue;
          auto const density = densities[i];
          if (colmaker_train_param_.NeedForwardSearch(
                  param_.default_direction, density, ind)) {
            this->EnumerateSplit(col.data(), col.data() + col.size(), +1, fid,
                                 gpair, stemp_[tid]);
          }
          if (colmaker_train_param_.NeedBackwardSearch(param_.default_direction)) {
            this->EnumerateSplit(col.data() + col.size() - 1, col.data() - 1,
                                 /*d_step=*/-1, fid, gpair, stemp_[tid]);
          }
        }
      }
    }
    // find splits at current level, do split per level
    void FindSplit(int depth, const std::vector<int> &qexpand,
                   const std::vector<GradientPair> &gpair, DMatrix *p_fmat,
                   RegTree *p_tree) {
      std::shared_ptr<HostDeviceVector<bst_feature_t>> feat_set = column_sampler_.GetFeatureSet(depth);
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) {
        this->UpdateSolution(batch, feat_set->HostVector(), gpair, p_fmat);
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      this->SyncBestSolution(qexpand);
      // get the best result, we can synchronize the solution
      for (int nid : qexpand) {
        NodeEntry const &e = nodes_statistics_[nid];
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > kRtEps) {
          bst_float left_leaf_weight =
              spliteval_->ComputeWeight(nid, e.best.left_sum) *
              param_.learning_rate;
          bst_float right_leaf_weight =
              spliteval_->ComputeWeight(nid, e.best.right_sum) *
              param_.learning_rate;
          // This marks the new nodes to be "fresh"
          p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                             e.best.DefaultLeft(), e.weight, left_leaf_weight,
                             right_leaf_weight, e.best.loss_chg,
                             e.stats.sum_hess, 0);
        } else {
          (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate);
        }
      }
    }
    // reset position of each data points after split is created in the tree
    inline void ResetPosition(const std::vector<int> &qexpand,
                              DMatrix* p_fmat,
                              const RegTree& tree) {
      // set the positions in the nondefault
      this->SetNonDefaultPosition(qexpand, p_fmat, tree);
      // set rest of instances to default position
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid,
      // so that they are ignored in future statistics collection
      const auto ndata = static_cast<bst_omp_uint>(p_fmat->Info().num_row_);

#pragma omp parallel for schedule(static)
      for (bst_omp_uint ridx = 0; ridx < ndata; ++ridx) {
        CHECK_LT(ridx, position_.size())
            << "ridx exceed bound " << "ridx="<<  ridx << " pos=" << position_.size();
        const int nid = this->DecodePosition(ridx);
        if (tree[nid].IsLeaf()) {
          // mark finish when it is not a fresh leaf
          if (tree[nid].RightChild() == -1) {
            position_[ridx] = ~nid;
          }
        } else {
          // push to default branch
          if (tree[nid].DefaultLeft()) {
            this->SetEncodePosition(ridx, tree[nid].LeftChild());
          } else {
            this->SetEncodePosition(ridx, tree[nid].RightChild());
          }
        }
      }
    }
    // customization part
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (int nid : qexpand) {
        NodeEntry &e = nodes_statistics_[nid];
        for (int tid = 0; tid < this->nthread_; ++tid) {
          e.best.Update(stemp_[tid][nid].best);
        }
      }
    }
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       DMatrix *p_fmat,
                                       const RegTree &tree) {
      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (int nid : qexpand) {
        if (!tree[nid].IsLeaf()) {
          fsplits.push_back(tree[nid].SplitIndex());
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>()) {
        for (auto fid : fsplits) {
          auto col = batch[fid];
          const auto ndata = static_cast<bst_omp_uint>(col.size());
#pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const int nid = this->DecodePosition(ridx);
            const bst_float fvalue = col[j].fvalue;
            // go back to parent, correct those who are not default
            if (!tree[nid].IsLeaf() && tree[nid].SplitIndex() == fid) {
              if (fvalue < tree[nid].SplitCond()) {
                this->SetEncodePosition(ridx, tree[nid].LeftChild());
              } else {
                this->SetEncodePosition(ridx, tree[nid].RightChild());
              }
            }
          }
        }
      }
    }
    // utils to get/set position, with encoded format
    // return decoded position
    inline int DecodePosition(bst_uint ridx) const {
      const int pid = position_[ridx];
      return pid < 0 ? ~pid : pid;
    }
    // encode the encoded position value for ridx
    inline void SetEncodePosition(bst_uint ridx, int nid) {
      if (position_[ridx] < 0) {
        position_[ridx] = ~nid;
      } else {
        position_[ridx] = nid;
      }
    }
    //  --data fields--
    const TrainParam& param_;
    const ColMakerTrainParam& colmaker_train_param_;
    // number of omp thread used during training
    const int nthread_;
    common::ColumnSampler column_sampler_;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position_;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector< std::vector<ThreadEntry> > stemp_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> nodes_statistics_;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
    // Evaluates splits and computes optimal weights for a given split
    std::unique_ptr<SplitEvaluator> spliteval_;
    // Implementation of interaction constraints.
    FeatureInteractionConstraintHost interaction_constraints_;
  };
};

XGBOOST_REGISTER_TREE_UPDATER(ColMaker, "grow_colmaker")
.describe("Grow tree with parallelization over columns.")
.set_body([]() {
    return new ColMaker();
  });
}  // namespace tree
}  // namespace xgboost
