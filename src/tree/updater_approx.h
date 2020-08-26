#ifndef XGBOOST_TREE_APPROX_H_
#define XGBOOST_TREE_APPROX_H_

#include "xgboost/tree_updater.h"
#include "xgboost/json.h"
#include "hist/param.h"
#include "constraints.h"
#include "driver.h"
#include "../common/random.h"
#include "hist/evaluate_splits.h"
#include "hist/row_partitioner.h"
#include "hist/evaluate_splits.h"
#include "param.h"

namespace xgboost {
namespace tree {
template <typename GradientSumT> class ApproxHistogramBuilder {
  using GradientPairT =
      std::conditional_t<std::is_same<GradientSumT, float>::value, GradientPair,
                         GradientPairPrecise>;

  common::HistCollection<GradientSumT> histograms_;
  common::ParallelGHistBuilder<GradientSumT> histogram_mapper_;
  common::GHistBuilder<GradientSumT> histogram_builder_;
  rabit::Reducer<GradientPairT, GradientPairT::Reduce> histogram_reducer_;

 public:
  void BuildNodeHistogram(const std::vector<GradientPair> &gpair,
                          common::RowSetCollection const& row_set_collection,
                          common::GHistIndexMatrix const &gidx,
                          bool is_dense,
                          std::vector<bst_node_t> const &nodes_to_build) {
    size_t n_nodes = nodes_to_build.size();
    CHECK_NE(n_nodes, 0);
    common::BlockedSpace2d space(n_nodes, [&](size_t nidx_in_set) {
      const int32_t nidx = nodes_to_build[nidx_in_set];
      return row_set_collection[nidx].Size();
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
      auto start_of_row_set = row_set_collection[nidx].begin;
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
  }

  void BuildHistogram(const std::vector<GradientPair> &gpair,
                      std::vector<LocalExpandEntry> candidates,
                      common::RowSetCollection const &row_indices,
                      bool is_dense, common::GHistIndexMatrix const &gidx,
                      RegTree const* p_tree) {
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

    this->BuildNodeHistogram(gpair, row_indices, gidx, is_dense, nodes_to_build);

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
  }

  auto const& Histograms() const { return histograms_; }

  ApproxHistogramBuilder() = default;

  explicit ApproxHistogramBuilder(size_t total_bins) {
    histograms_.Init(total_bins);
    histogram_mapper_.Init(total_bins);
    histogram_builder_ =
        common::GHistBuilder<GradientSumT>(omp_get_max_threads(), total_bins);
  }
};

class ApproxRowPartitioner {
  static constexpr size_t kPartitionBlockSize = 2048;
  PartitionBuilder<kPartitionBlockSize> partition_builder_;
  common::RowSetCollection row_set_collection_;

  static auto SearchCutValue(bst_row_t ridx, bst_feature_t fidx,
                             common::GHistIndexMatrix const &index,
                             std::vector<uint32_t> const &cut_ptrs,
                             std::vector<float> const &cut_values) {
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
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return cut_values[gidx];
  }

 public:
  void UpdatePosition(common::GHistIndexMatrix const &index,
                      std::vector<LocalExpandEntry> const &candidates,
                      RegTree const *p_tree) {
      size_t n_nodes = candidates.size();

    auto const& cut_values = index.cut.Values();
    auto const& cut_ptrs = index.cut.Ptrs();

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
                auto cut_value = SearchCutValue(row_id, fidx, index, cut_ptrs, cut_values);
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
  }

  auto const& Partitions() const { return row_set_collection_; }
  auto operator[](bst_node_t nidx) { return row_set_collection_[nidx]; }
  size_t Size() const {
    return std::distance(row_set_collection_.begin(),
                         row_set_collection_.end());
  }

  ApproxRowPartitioner() = default;
  explicit ApproxRowPartitioner(bst_row_t num_row) {
    row_set_collection_.Clear();
    auto p_positions = row_set_collection_.Data();
    p_positions->resize(num_row);
    std::iota(p_positions->begin(), p_positions->end(), 0);
    row_set_collection_.Init();
  }
};

struct NodeEntry {
  /*! \brief statics for node entry */
  GradStats stats;
  /*! \brief loss of this node, without split */
  bst_float root_gain;
};

template <typename GradientSumT> class ApproxEvaluator {
  TrainParam param_;
  common::ColumnSampler column_sampler_;
  TreeEvaluator tree_evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  std::vector<NodeEntry> snode_;

 public:
  void EvaluateSplits(const common::HistCollection<GradientSumT> &hist,
                      common::GHistIndexMatrix const &gidx, const RegTree &tree,
                      std::vector<LocalExpandEntry *> entries) {
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
    auto evaluator = tree_evaluator_.GetEvaluator();

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
  }

  void
  ApplyTreeSplit(LocalExpandEntry candidate, TrainParam param, RegTree *p_tree) {
    auto evaluator = tree_evaluator_.GetEvaluator();
    RegTree &tree = *p_tree;

    GradStats parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    auto base_weight =
        evaluator.CalcWeight(candidate.nid, param, GradStats{parent_sum});

    auto left_weight =
        evaluator.CalcWeight(candidate.nid, param,
                             GradStats{candidate.split.left_sum}) *
        param.learning_rate;
    auto right_weight =
        evaluator.CalcWeight(candidate.nid, param,
                             GradStats{candidate.split.right_sum}) *
        param.learning_rate;

    tree.ExpandNode(candidate.nid, candidate.split.SplitIndex(),
                    candidate.split.split_value, candidate.split.DefaultLeft(),
                    base_weight, left_weight, right_weight,
                    candidate.split.loss_chg, parent_sum.GetHess(),
                    candidate.split.left_sum.GetHess(),
                    candidate.split.right_sum.GetHess());

    // Set up child constraints
    auto left_child = tree[candidate.nid].LeftChild();
    auto right_child = tree[candidate.nid].RightChild();
    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child,
                             tree[candidate.nid].SplitIndex(), left_weight,
                             right_weight);

    auto max_node = std::max(left_child, tree[candidate.nid].RightChild());
    max_node = std::max(candidate.nid, max_node);
    snode_.resize(tree.GetNodes().size());
    snode_.at(left_child).stats = candidate.split.left_sum;
    snode_.at(left_child).root_gain = evaluator.CalcGain(
        candidate.nid, param, GradStats{candidate.split.left_sum});
    snode_.at(right_child).stats = candidate.split.right_sum;
    snode_.at(right_child).root_gain = evaluator.CalcGain(
        candidate.nid, param, GradStats{candidate.split.right_sum});

    interaction_constraints_.Split(candidate.nid,
                                   tree[candidate.nid].SplitIndex(), left_child,
                                   right_child);
  }

  auto GetEvaluator() const { return tree_evaluator_.GetEvaluator(); }
  auto const& Stats() const { return snode_; }

  float InitRoot(GradStats const& root_sum) {
    snode_.resize(1);
    auto root_evaluator = tree_evaluator_.GetEvaluator();

    snode_[0].stats = GradStats{root_sum.GetGrad(), root_sum.GetHess()};
    snode_[0].root_gain = root_evaluator.CalcGain(RegTree::kRoot, param_,
                                                  GradStats{snode_[0].stats});
    auto weight = root_evaluator.CalcWeight(RegTree::kRoot, param_,
                                            GradStats{snode_[0].stats});
    return weight;
  }

  ApproxEvaluator() = default;
  explicit ApproxEvaluator(TrainParam param, MetaInfo const &info)
      : param_{std::move(param)}, tree_evaluator_{
                                      param,
                                      static_cast<bst_feature_t>(info.num_col_),
                                      GenericParameter::kCpuId} {
    column_sampler_.Init(info.num_col_, info.feature_weigths.HostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, false);
  }
};

inline void ApproxLazyInitData(TrainParam const &param,
                               HostDeviceVector<GradientPair> *gpair,
                               DMatrix *m, DMatrix *cached,
                               std::vector<GradientPair> *sampled,
                               std::vector<bst_row_t> *p_columns_size) {
  auto const &info = m->Info();
  auto& columns_size = *p_columns_size;
  if (columns_size.empty() || cached != m) {
    columns_size.resize(info.num_col_, 0);
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
        columns_size[i] += entries_per_column[i];
      }
    }
  }

  auto const &h_gpair = gpair->HostVector();
  sampled->resize(h_gpair.size());
  std::copy(h_gpair.cbegin(), h_gpair.cend(), sampled->begin());
  auto &rnd = common::GlobalRandom();
  if (param.subsample != 1.0) {
    CHECK(param.sampling_method != TrainParam::kGradientBased)
        << "Gradient based sampling is not supported for approx tree method.";
    std::bernoulli_distribution coin_flip(param.subsample);
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
}      // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_APPROX_H_
