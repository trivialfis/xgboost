#include <future>
#include <vector>

#include "xgboost/tree_updater.h"
#include "xgboost/base.h"
#include <parallel/algorithm>
#include "xgboost/json.h"
#include "hist/row_partitioner.h"
#include "hist/evaluate_splits.h"
#include "hist/param.h"
#include "constraints.h"
#include "driver.h"
#include "param.h"
#include "../common/random.h"
#include "updater_approx.h"

namespace xgboost {
namespace tree {

template <typename GradientSumT> class GloablApproxBuilder {
 protected:
  using GradientPairT =
      std::conditional_t<std::is_same<GradientSumT, float>::value, GradientPair,
                         GradientPairPrecise>;

  TrainParam param_;
  CPUHistMakerTrainParam hist_param_;

  ApproxEvaluator<GradientSumT> evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;

  ApproxHistogramBuilder<GradientSumT> histogram_builder_;

  ApproxRowPartitioner partitioner_;
  std::vector<NodeEntry> snode_;
  RegTree* p_last_tree_ {nullptr};
  common::Monitor* monitor_;

 public:
  void InitData(DMatrix *m, std::vector<GradientPair> const &gpair,
                common::GHistIndexMatrix const &index) {
    monitor_->Start(__func__);
    auto const &info = m->Info();
    partitioner_ = ApproxRowPartitioner(info.num_row_);
    histogram_builder_ = ApproxHistogramBuilder<GradientSumT>(index.cut.TotalBins());
    monitor_->Stop(__func__);
  }

  LocalExpandEntry InitRoot(common::GHistIndexMatrix const &m,
                            std::vector<GradientPair> const &gpair,
                            RegTree *p_tree) {
    monitor_->Start(__func__);
    LocalExpandEntry best;
    best.nid = RegTree::kRoot;
    best.depth = 0;
    GradStats root_sum;
    for (auto const& g : gpair) {
      root_sum.Add(g);
    }
    rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double *>(&root_sum), 2);

    this->BuildNodeHistogram(gpair, m, m.IsDense(), {RegTree::kRoot});

    auto weight = evaluator_.InitRoot(root_sum);
    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    auto const& histograms = histogram_builder_.Histograms();
    this->EvaluateSplits(histograms, m, *p_tree, {&best});
    monitor_->Stop(__func__);

    return best;
  }

  void EvaluateSplits(const common::HistCollection<GradientSumT> &hist,
                      common::GHistIndexMatrix const &gidx, const RegTree &tree,
                      std::vector<LocalExpandEntry *> entries) {
    monitor_->Start(__func__);
    evaluator_.EvaluateSplits(hist, gidx, tree, entries);
    monitor_->Stop(__func__);
  }

  void ApplySplit(LocalExpandEntry candidate, RegTree *p_tree) {
    monitor_->Start(__func__);
    evaluator_.ApplyTreeSplit(candidate, param_, p_tree);
    monitor_->Stop(__func__);
  }

  void UpdatePosition(common::GHistIndexMatrix const &index,
                      std::vector<LocalExpandEntry> const &candidates,
                      RegTree const *p_tree) {
    monitor_->Start(__func__);
    partitioner_.UpdatePosition(index, candidates, p_tree);
    monitor_->Stop(__func__);
  }

  void UpdatePredictionCache(const DMatrix *data,
                             HostDeviceVector<bst_float> *p_out_preds) {
    monitor_->Start(__func__);
    // Caching prediction seems redundant for approx tree method, as sketching takes up
    // majority of training time.
    std::vector<bst_float>& out_preds = p_out_preds->HostVector();
    CHECK_EQ(out_preds.size(), data->Info().num_row_);
    CHECK(p_last_tree_);

    size_t n_nodes = p_last_tree_->GetNodes().size();
    CHECK_EQ(partitioner_.Size(), n_nodes);
    common::BlockedSpace2d space(
        n_nodes, [&](size_t node) { return partitioner_[node].Size(); },
        1024);

    auto evaluator = evaluator_.GetEvaluator();
    auto const& tree = *p_last_tree_;
    common::ParallelFor2d(
        space, omp_get_max_threads(), [&](size_t nidx, common::Range1d r) {
          if (tree[nidx].IsLeaf()) {
            const auto rowset = partitioner_[nidx];
            auto const &stats = snode_.at(nidx);
            auto leaf_value =
                evaluator.CalcWeight(nidx, param_, GradStats{stats.stats}) *
                param_.learning_rate;
            for (const size_t *it = rowset.begin + r.begin();
                 it < rowset.begin + r.end(); ++it) {
              out_preds[*it] += leaf_value;
            }
          }
        });
    monitor_->Stop(__func__);
  }

  void BuildNodeHistogram(const std::vector<GradientPair> &gpair,
                          common::GHistIndexMatrix const &gidx,
                          bool is_dense,
                          std::vector<bst_node_t> const &nodes_to_build) {
    monitor_->Start(__func__);
    histogram_builder_.BuildNodeHistogram(gpair, partitioner_.Partitions(), gidx,
                                          is_dense, nodes_to_build);
    monitor_->Stop(__func__);
  }

  void BuildHistogram(const std::vector<GradientPair> &gpair,
                      std::vector<LocalExpandEntry> candidates,
                      common::RowSetCollection const &row_indices,
                      bool is_dense, common::GHistIndexMatrix const &gidx,
                      RegTree const* p_tree) {
    monitor_->Start(__func__);
    histogram_builder_.BuildHistogram(gpair, candidates, row_indices, is_dense,
                                      gidx, p_tree);
    monitor_->Stop(__func__);
  }

 public:
  explicit GloablApproxBuilder(TrainParam param, CPUHistMakerTrainParam hparam,
                               MetaInfo const &info, common::Monitor *monitor)
      : param_{std::move(param)}, hist_param_{hparam},
        evaluator_{param_, info}, monitor_{monitor} {}

  void UpdateTree(RegTree *p_tree, DMatrix *m,
                  std::vector<GradientPair> const &gpair,
                  common::GHistIndexMatrix const &index) {
    p_last_tree_ = p_tree;
    this->InitData(m, gpair, index);

    auto &tree = *p_tree;

    DriverContainer<LocalExpandEntry> driver(
        static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));

    driver.Push({this->InitRoot(index, gpair, p_tree)});
    auto num_leaves = 1;
    auto expand_set = driver.Pop();

    while (!expand_set.empty()) {
      std::vector<LocalExpandEntry> new_candidates(expand_set.size() * 2);
      // candidates that can further splited.
      std::vector<LocalExpandEntry> valid_candidates;
      // candidaates that can be applied.
      std::vector<LocalExpandEntry> applid;
      std::vector<size_t> nidx_set;
      for (size_t i = 0; i < expand_set.size(); ++i) {
        auto candidate = expand_set[i];
        if (!candidate.IsValid(param_, num_leaves)) {
          continue;
        }
        this->ApplySplit(candidate, p_tree);
        applid.push_back(candidate);
        num_leaves++;
        int left_child_nidx = tree[candidate.nid].LeftChild();
        if (LocalExpandEntry::ChildIsValid(param_, p_tree->GetDepth(left_child_nidx),
                                           num_leaves)) {
          valid_candidates.emplace_back(candidate);
          nidx_set.emplace_back(i);
        } else {
          new_candidates[i * 2] = LocalExpandEntry();
          new_candidates[i * 2 + 1] = LocalExpandEntry();
        }
      }
      this->UpdatePosition(index, applid, p_tree);

      if (!valid_candidates.empty()) {
        this->BuildHistogram(gpair, valid_candidates, partitioner_.Partitions(),
                             m->IsDense(), index, p_tree);
        std::vector<LocalExpandEntry*> best_splits;
        std::vector<size_t> new_candidates_pos;
        for (size_t c = 0; c < valid_candidates.size(); ++c) {
          auto i = nidx_set[c];
          auto candidate = valid_candidates[c];
          int left_child_nidx = tree[candidate.nid].LeftChild();
          int right_child_nidx = tree[candidate.nid].RightChild();
          LocalExpandEntry l_best{
              left_child_nidx, tree.GetDepth(left_child_nidx), {}};
          LocalExpandEntry r_best{
              right_child_nidx, tree.GetDepth(right_child_nidx), {}};
          new_candidates[i * 2] = l_best;
          new_candidates[i * 2 + 1] = r_best;
          best_splits.push_back(&new_candidates[i * 2]);
          best_splits.push_back(&new_candidates[i * 2 + 1]);
        }
        auto const& histograms = histogram_builder_.Histograms();
        this->EvaluateSplits(histograms, index, tree, best_splits);
      }
      driver.Push(new_candidates.begin(), new_candidates.end());
      expand_set = driver.Pop();
    }
  }
};

class GlobalApproxUpdater : public TreeUpdater {
  TrainParam param_;
  common::Monitor monitor_;
  CPUHistMakerTrainParam hist_param_;
  std::vector<bst_row_t> columns_size_;

  std::unique_ptr<GloablApproxBuilder<float>> f32_impl_;
  std::unique_ptr<GloablApproxBuilder<double>> f64_impl_;
  DMatrix* cached_ { nullptr };

 public:
  GlobalApproxUpdater() {
    monitor_.Init(__func__);
  }

  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    hist_param_.UpdateAllowUnknown(args);
  }
  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    FromJson(config.at("hist_param"), &this->hist_param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["hist_param"] = ToJson(hist_param_);
  }

  char const *Name() const override { return "grow_global_approx_histmaker"; }

  void InitData(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
                std::vector<GradientPair> *sampled) {
    auto const &info = m->Info();
    if (columns_size_.empty() || cached_ != m) {
      cached_ = m;
      columns_size_.resize(info.num_col_, 0);
      const auto threads = omp_get_max_threads();
      std::vector<std::vector<bst_row_t>> column_sizes(threads);
      for (auto &column : column_sizes) {
        column.resize(info.num_col_, 0);
      }
      for (auto const &page : m->GetBatches<SparsePage>()) {
        auto const &entries_per_column =
            common::HostSketchContainer::CalcColumnSize(page, info.num_col_,
                                                        threads);
        for (size_t i = 0; i < entries_per_column.size(); ++i) {
          columns_size_[i] += entries_per_column[i];
        }
      }
    }

    auto const &h_gpair = gpair->HostVector();
    sampled->resize(h_gpair.size());
    std::copy(h_gpair.cbegin(), h_gpair.cend(), sampled->begin());
    auto &rnd = common::GlobalRandom();
    if (param_.subsample != 1.0) {
      CHECK(param_.sampling_method != TrainParam::kGradientBased)
          << "Gradient based sampling is not supported for approx tree method.";
      std::bernoulli_distribution coin_flip(param_.subsample);
      std::transform(sampled->begin(), sampled->end(), sampled->begin(),
                     [&](GradientPair &g) {
                       if (coin_flip(rnd)) {
                         return g;
                       } else {
                         return GradientPair{};
                       }
                     });
    }
  }

  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
              const std::vector<RegTree *> &trees) override {
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();

    if (hist_param_.single_precision_histogram) {
      f32_impl_ = std::make_unique<GloablApproxBuilder<float>>(
          param_, hist_param_, m->Info(), &monitor_);
    } else {
      f64_impl_ = std::make_unique<GloablApproxBuilder<double>>(
          param_, hist_param_, m->Info(), &monitor_);
    }

    CHECK(!param_.enable_feature_grouping)
        << "Feature grouping is not implemented for approx.";

    std::vector<GradientPair> h_gpair;
    this->InitData(gpair, m, &h_gpair);
    auto const &info = m->Info();

    common::HistogramCuts cuts;
    std::vector<float> hessians(h_gpair.size());
    std::transform(h_gpair.cbegin(), h_gpair.cend(), hessians.begin(),
                   [](auto const &g) { return g.GetHess(); });
    common::HostSketchContainer container(columns_size_, param_.max_bin, false);
    if (m->IsDense()) {
      // This is actually twice slower, but removes a copy of data.
      monitor_.Start("Dense Sketch");
      for (auto const& page : m->GetBatches<SparsePage>()) {
        container.PushRowPage(page, info);
      }
      monitor_.Stop("Dense Sketch");
    } else {
      m->GetBatches<SortedCSCPage>();
      monitor_.Start("Sparse Sketch");
      for (auto const &page : m->GetBatches<SortedCSCPage>()) {
        container.PushSortedCSC(page, info, hessians);
      }
      monitor_.Stop("Sparse Sketch");
    }
    container.MakeCuts(&cuts);

    monitor_.Start("GHistIndexMatrix");
    common::GHistIndexMatrix gidx;
    gidx.Init(m, cuts, param_.max_bin);
    monitor_.Stop("GHistIndexMatrix");

    for (auto p_tree : trees) {
      if (hist_param_.single_precision_histogram) {
        this->f32_impl_->UpdateTree(p_tree, m, h_gpair, gidx);
      } else {
        this->f64_impl_->UpdateTree(p_tree, m, h_gpair, gidx);
      }
    }
    param_.learning_rate = lr;
  }

  bool
  UpdatePredictionCache(const DMatrix *data,
                        HostDeviceVector<bst_float> *p_out_preds) override {
    if (data != cached_) { return false; }

    if (hist_param_.single_precision_histogram) {
      this->f32_impl_->UpdatePredictionCache(data, p_out_preds);
    } else {
      this->f64_impl_->UpdatePredictionCache(data, p_out_preds);
    }
    return true;
  }
};

#if !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(grow_global_approx_histmaker);
XGBOOST_REGISTER_TREE_UPDATER(GlobalHistMaker, "grow_global_approx_histmaker")
    .describe("Tree constructor that uses approximate histogram construction "
              "for each node.")
    .set_body([]() { return new GlobalApproxUpdater(); });
#endif  // !defined(GTEST_TEST)
}  // namespace tree
}  // namespace xgboost
