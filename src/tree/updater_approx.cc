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

namespace xgboost {
namespace tree {

template <typename GradientSumT> class GloablApproxBuilder {
 protected:
  using GradientPairT =
      std::conditional_t<std::is_same<GradientSumT, float>::value, GradientPair,
                         GradientPairPrecise>;

  TrainParam param_;
  CPUHistMakerTrainParam hist_param_;

  TreeEvaluator evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  common::ColumnSampler column_sampler_;
  std::vector<int> monotonic_constraint_;

  common::GHistBuilder<GradientSumT> histogram_builder_;

  static constexpr size_t kPartitionBlockSize = 2048;
  PartitionBuilder<kPartitionBlockSize> partition_builder_;
  common::RowSetCollection row_set_collection_;

  common::HistCollection<GradientSumT> histograms_;
  rabit::Reducer<GradientPairT, GradientPairT::Reduce> histogram_reducer_;
  common::ParallelGHistBuilder<GradientSumT> histogram_mapper_;

  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
  };
  std::vector<NodeEntry> snode_;
  RegTree* p_last_tree_ {nullptr};
  common::Monitor* monitor_;

 public:
  void InitData(DMatrix *m, std::vector<GradientPair> const &gpair,
                common::GHistIndexMatrix const &index) {
    monitor_->Start(__func__);
    auto const &info = m->Info();

    row_set_collection_.Clear();
    auto p_positions = row_set_collection_.Data();
    p_positions->resize(info.num_row_);
    std::iota(p_positions->begin(), p_positions->end(), 0);
    row_set_collection_.Init();

    column_sampler_.Init(info.num_col_, m->Info().feature_weigths.HostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, false);

    if (!param_.monotone_constraints.empty()) {
      monotonic_constraint_ = param_.monotone_constraints;
    } else {
      monotonic_constraint_.resize(info.num_col_, 0);
    }

    histograms_.Init(index.cut.TotalBins());
    histogram_mapper_.Init(index.cut.TotalBins());
    histogram_builder_ =
        common::GHistBuilder<GradientSumT>(omp_get_max_threads(), index.cut.TotalBins());
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

    snode_.resize(p_tree->GetNodes().size());
    CHECK_EQ(snode_.size(), 1);
    auto root_evaluator = evaluator_.GetEvaluator();

    snode_[0].stats = GradStats{root_sum.GetGrad(), root_sum.GetHess()};
    snode_[0].root_gain = root_evaluator.CalcGain(RegTree::kRoot, param_,
                                                  GradStats{snode_[0].stats});
    auto weight = root_evaluator.CalcWeight(RegTree::kRoot, param_,
                                            GradStats{snode_[0].stats});


    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    this->EvaluateSplits(histograms_, m, *p_tree, {&best});
    monitor_->Stop(__func__);

    return best;
  }

  void EvaluateSplits(const common::HistCollection<GradientSumT> &hist,
                      common::GHistIndexMatrix const &gidx, const RegTree &tree,
                      std::vector<LocalExpandEntry *> entries) {
    monitor_->Start(__func__);
    const size_t grain_size = std::max<size_t>(
        1,
        column_sampler_.GetFeatureSet(tree.GetDepth(entries[0]->nid))->Size() /
        omp_get_max_threads());

    // All nodes are on the same level, so we can store the shared ptr.
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(
        entries.size());
    for (size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      auto nidx = entries[nidx_in_set]->nid;
      features[nidx_in_set] = column_sampler_.GetFeatureSet(tree.GetDepth(nidx));
    }
    common::BlockedSpace2d space(entries.size(), [&](size_t nidx_in_set) {
      return features[nidx_in_set]->Size();
    }, grain_size);

    auto num_threads = omp_get_max_threads();
    std::vector<LocalExpandEntry> tloc_candidates(omp_get_max_threads() * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      for (decltype(num_threads) j = 0; j < num_threads; ++j) {
        tloc_candidates[i * num_threads + j] = *entries[i];
      }
    }
    auto evaluator = evaluator_.GetEvaluator();

    common::ParallelFor2d(space, num_threads, [&](size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[num_threads * nidx_in_set + tidx];
      auto best = &entry->split;
      auto nidx = entry->nid;
      auto histogram = hist[nidx];
      auto features_set = features[nidx_in_set]->ConstHostSpan();
      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        if (interaction_constraints_.Query(nidx, fidx)) {
          auto grad_stats = EnumerateSplit<common::GHistRow<GradientSumT>,
                                           NodeEntry, SplitEntry, +1>(
              gidx, histogram, snode_[nidx], best, nidx, fidx, param_,
              evaluator);
          if (SplitContainsMissingValues(grad_stats, snode_[nidx])) {
            EnumerateSplit<common::GHistRow<GradientSumT>, NodeEntry,
                           SplitEntry, -1>(gidx, histogram, snode_[nidx], best,
                                           nidx, fidx, param_, evaluator);
          }
        }
      }
    });

    for (unsigned nidx_in_set = 0; nidx_in_set < entries.size();
         ++nidx_in_set) {
      for (auto tidx = 0; tidx < num_threads; ++tidx) {
        entries[nidx_in_set]->split.Update(
            tloc_candidates[num_threads * nidx_in_set + tidx].split);
      }
    }
    monitor_->Stop(__func__);
  }

  void ApplySplit(LocalExpandEntry candidate, RegTree *p_tree) {
    monitor_->Start(__func__);
    RegTree &tree = *p_tree;

    GradStats parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    auto tree_evalator = evaluator_.GetEvaluator();
    auto base_weight =
        tree_evalator.CalcWeight(candidate.nid, param_, GradStats{parent_sum});

    auto left_weight =
        tree_evalator.CalcWeight(candidate.nid, param_,
                                 GradStats{candidate.split.left_sum}) *
        param_.learning_rate;
    auto right_weight =
        tree_evalator.CalcWeight(candidate.nid, param_,
                                 GradStats{candidate.split.right_sum}) *
        param_.learning_rate;

    tree.ExpandNode(candidate.nid, candidate.split.SplitIndex(),
                    candidate.split.split_value, candidate.split.DefaultLeft(),
                    base_weight, left_weight, right_weight,
                    candidate.split.loss_chg, parent_sum.GetHess(),
                    candidate.split.left_sum.GetHess(),
                    candidate.split.right_sum.GetHess());

    // Set up child constraints
    auto left_child = tree[candidate.nid].LeftChild();
    auto right_child = tree[candidate.nid].RightChild();
    evaluator_.AddSplit(candidate.nid, left_child, right_child,
                        tree[candidate.nid].SplitIndex(), left_weight,
                        right_weight);

    auto max_node = std::max(left_child, tree[candidate.nid].RightChild());
    max_node = std::max(candidate.nid, max_node);
    snode_.resize(tree.GetNodes().size());
    snode_.at(left_child).stats = candidate.split.left_sum;
    snode_.at(left_child).root_gain = tree_evalator.CalcGain(
        candidate.nid, param_, GradStats{candidate.split.left_sum});
    snode_.at(right_child).stats = candidate.split.right_sum;
    snode_.at(right_child).root_gain = tree_evalator.CalcGain(
        candidate.nid, param_, GradStats{candidate.split.right_sum});

    interaction_constraints_.Split(candidate.nid,
                                   tree[candidate.nid].SplitIndex(), left_child,
                                   right_child);
    monitor_->Stop(__func__);
  }

  void UpdatePosition(common::GHistIndexMatrix const &index,
                      std::vector<LocalExpandEntry> const &candidates,
                      RegTree const *p_tree) {
    monitor_->Start(__func__);
    size_t n_nodes = candidates.size();

    auto const& cut_values = index.cut.Values();
    auto const& cut_ptrs = index.cut.Ptrs();
    auto search_cut_value = [&](bst_row_t ridx, bst_feature_t fidx) {
      int32_t gidx = -1;
      if (index.IsDense()) {
        gidx = index.index[index.row_ptr[ridx] + fidx];
      } else {
        auto begin = index.row_ptr[ridx];
        auto end = index.row_ptr[ridx + 1];
        auto f_begin = cut_ptrs[fidx];
        auto f_end = cut_ptrs[fidx + 1];
        gidx = common::BinarySearchBin(begin, end, index.index, f_begin, f_end);
      }
      if (gidx == -1){
        return std::numeric_limits<float>::quiet_NaN();
      }
      return cut_values[gidx];
    };

    common::BlockedSpace2d space{n_nodes,
                                 [&](size_t node_in_set) {
                                   auto candidate = candidates[node_in_set];
                                   int32_t nid = candidate.nid;
                                   return row_set_collection_[nid].Size();
                                 },
                                 kPartitionBlockSize};
    partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
      auto candidate = candidates[node_in_set];
      const int32_t nid = candidate.nid;
      const size_t size = row_set_collection_[nid].Size();
      const size_t n_tasks =
          size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
      return n_tasks;
    });
    auto threads = omp_get_max_threads();
    common::ParallelFor2d(
        space, threads, [&](size_t node_in_set, common::Range1d r) {
          auto candidate = candidates[node_in_set];
          const int32_t nid = candidate.nid;
          auto fidx = candidate.split.SplitIndex();
          partition_builder_.template PartitionRange(
              node_in_set, nid, r, fidx, &row_set_collection_,
              [&](size_t row_id) {
                auto cut_value = search_cut_value(row_id, fidx);
                if (std::isnan(cut_value)) {
                  return candidate.split.DefaultLeft();
                }
                return cut_value <= candidate.split.split_value;
              });
        });

    partition_builder_.CalculateRowOffsets();
    common::ParallelFor2d(
        space, threads, [&](size_t node_in_set, common::Range1d r) {
          auto candidate = candidates[node_in_set];
          const int32_t nid = candidate.nid;
          partition_builder_.MergeToArray(
              node_in_set, r.begin(),
              const_cast<size_t *>(row_set_collection_[nid].begin));
        });
    for (size_t i = 0; i < candidates.size(); ++i) {
      auto const& candidate = candidates[i];
      auto nidx = candidate.nid;
      auto n_left = partition_builder_.GetNLeftElems(i);
      auto n_right = partition_builder_.GetNRightElems(i);
      CHECK_EQ(n_left + n_right, row_set_collection_[nidx].Size());
      bst_node_t left_nidx = (*p_tree)[nidx].LeftChild();
      bst_node_t right_nidx = (*p_tree)[nidx].RightChild();
      row_set_collection_.AddSplit(nidx, left_nidx, right_nidx, n_left,
                                   n_right);
    }
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
    CHECK_EQ(std::distance(row_set_collection_.begin(), row_set_collection_.end()), n_nodes);
    common::BlockedSpace2d space(
        n_nodes, [&](size_t node) { return row_set_collection_[node].Size(); },
        1024);

    auto evaluator = evaluator_.GetEvaluator();
    auto const& tree = *p_last_tree_;
    common::ParallelFor2d(
        space, omp_get_max_threads(), [&](size_t nidx, common::Range1d r) {
          if (tree[nidx].IsLeaf()) {
            const auto rowset = row_set_collection_[nidx];
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
    size_t n_nodes = nodes_to_build.size();
    CHECK_NE(n_nodes, 0);
    common::BlockedSpace2d space(n_nodes, [&](size_t nidx_in_set) {
      const int32_t nidx = nodes_to_build[nidx_in_set];
      return row_set_collection_[nidx].Size();
    }, 256);

    for (auto nidx : nodes_to_build) {
      histograms_.AddHistRow(nidx);
    }

    std::vector<common::GHistRow<GradientSumT>> target_hists(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
      const int32_t nid = nodes_to_build[i];
      target_hists[i] = histograms_[nid];
    }
    histogram_mapper_.Reset(omp_get_max_threads(), n_nodes, space, target_hists);
    common::ParallelFor2d(space, omp_get_max_threads(), [&](size_t nidx_in_set, common::Range1d r) {
      auto const tidx = omp_get_thread_num();
      auto const nidx = nodes_to_build[nidx_in_set];
      auto start_of_row_set = row_set_collection_[nidx].begin;
      auto rid_set = common::RowSetCollection::Elem(
          start_of_row_set + r.begin(), start_of_row_set + r.end(), nidx);
      auto hist = histogram_mapper_.GetInitializedHist(tidx, nidx_in_set);
      histogram_builder_.BuildHist(gpair, rid_set, gidx, hist, is_dense);
    });

    const uint32_t total_bins = histogram_builder_.GetNumBins();
    common::BlockedSpace2d reduction_space(
        n_nodes, [&](size_t node) { return total_bins; }, 1024);
    common::ParallelFor2d(reduction_space, omp_get_max_threads(),
                          [&](size_t nidx_in_set, common::Range1d r) {
                            auto nidx = nodes_to_build[nidx_in_set];
                            auto hist = this->histograms_[nidx];
                            histogram_mapper_.ReduceHist(nidx_in_set, r.begin(),
                                                         r.end());
                          });

    if (!rabit::IsDistributed()) {
      monitor_->Stop(__func__);
      return;
    }

    std::vector<GradientPairT> reduction(total_bins * n_nodes);
    common::Span<GradientPairT> s_reduction(reduction);
    common::ParallelFor2d(reduction_space, omp_get_max_threads(),
                          [&](size_t nidx_in_set, common::Range1d r) {
                            auto nidx = nodes_to_build[nidx_in_set];
                            auto hist = this->histograms_[nidx];
                            auto dst = s_reduction.subspan(
                                nidx_in_set * total_bins, total_bins);
                            CHECK_EQ(hist.size(), dst.size());
                            common::CopyHist(dst, hist, r.begin(), r.end());
                          });
    CHECK_NE(total_bins * n_nodes, 0);
    histogram_reducer_.Allreduce(s_reduction.data(), total_bins * n_nodes);
    common::ParallelFor2d(reduction_space, omp_get_max_threads(),
                          [&](size_t nidx_in_set, common::Range1d r) {
                            auto nidx = nodes_to_build[nidx_in_set];
                            auto src = s_reduction.subspan(
                                nidx_in_set * total_bins, total_bins);
                            auto hist = this->histograms_[nidx];
                            common::CopyHist(hist, src, r.begin(), r.end());
                          });
    monitor_->Stop(__func__);
  }

  void BuildHistogram(const std::vector<GradientPair> &gpair,
                      std::vector<LocalExpandEntry> candidates,
                      common::RowSetCollection const &row_indices,
                      bool is_dense, common::GHistIndexMatrix const &gidx,
                      RegTree const* p_tree) {
    monitor_->Start(__func__);
    CHECK_NE(candidates.size(), 0);
    auto const& tree = *p_tree;
    std::vector<bst_node_t> nodes_to_build;
    std::vector<bst_node_t> nodes_to_subtract;
    for (auto const& candidate : candidates) {
      auto parent_hist = histograms_[candidate.nid];
      int left_nidx = tree[candidate.nid].LeftChild();
      int right_nidx = tree[candidate.nid].RightChild();
      auto build_hist_nidx = left_nidx;
      auto subtraction_trick_nidx = right_nidx;
      bool fewer_right = candidate.split.right_sum.GetHess() <
                         candidate.split.left_sum.GetHess();
      if (fewer_right) {
        std::swap(build_hist_nidx, subtraction_trick_nidx);
      }
      nodes_to_build.push_back(build_hist_nidx);
      nodes_to_subtract.push_back(subtraction_trick_nidx);
    }
    CHECK_EQ(nodes_to_build.size(), nodes_to_subtract.size());
    CHECK_EQ(nodes_to_build.size(), candidates.size());

    this->BuildNodeHistogram(gpair, gidx, is_dense, nodes_to_build);

    for (auto nidx : nodes_to_subtract) {
      histograms_.AddHistRow(nidx);
    }
    common::BlockedSpace2d reduction_space(
        nodes_to_subtract.size(), [&](size_t node) { return gidx.cut.TotalBins(); }, 1024);
    common::ParallelFor2d(reduction_space, omp_get_max_threads(),
                          [&](size_t nidx_in_set, common::Range1d r) {
                            auto const& candidate = candidates[nidx_in_set];
                            auto build_hist_nidx = nodes_to_build[nidx_in_set];
                            auto subtraction_trick_nidx =
                                nodes_to_subtract[nidx_in_set];

                            CHECK_EQ(histograms_[subtraction_trick_nidx].size(),
                                     histograms_[candidate.nid].size());
                            CHECK_EQ(histograms_[candidate.nid].size(),
                                     histograms_[build_hist_nidx].size());
                            common::SubtractionHist(
                                histograms_[subtraction_trick_nidx],
                                histograms_[candidate.nid],
                                histograms_[build_hist_nidx], r.begin(),
                                r.end());
                          });
    monitor_->Stop(__func__);
  }

 public:
  explicit GloablApproxBuilder(TrainParam param, CPUHistMakerTrainParam hparam,
                               bst_feature_t n_features, common::Monitor* monitor)
      : param_{param}, hist_param_{hparam},
        evaluator_(param, n_features, GenericParameter::kCpuId), monitor_{monitor} {}

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
        this->BuildHistogram(gpair, valid_candidates, row_set_collection_,
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
        this->EvaluateSplits(histograms_, index, tree, best_splits);
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
          param_, hist_param_, m->Info().num_col_, &monitor_);
    } else {
      f64_impl_ = std::make_unique<GloablApproxBuilder<double>>(
          param_, hist_param_, m->Info().num_col_, &monitor_);
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
