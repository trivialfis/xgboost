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

  common::Monitor* monitor_;
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
    monitor_->Start(__func__);
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
    monitor_->Stop(__func__);
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

struct NodeEntry {
  /*! \brief statics for node entry */
  GradStats stats;
  /*! \brief loss of this node, without split */
  bst_float root_gain;
};

inline void ApplyTreeSplit(
    LocalExpandEntry candidate, TrainParam param, RegTree *p_tree,
    std::vector<NodeEntry> *p_snode, TreeEvaluator *tree_evaluator,
    FeatureInteractionConstraintHost *p_interaction_constraints) {
  auto &snode = *p_snode;
  auto &interaction_constraints = *p_interaction_constraints;
  auto evaluator = tree_evaluator->GetEvaluator();
  RegTree &tree = *p_tree;

  GradStats parent_sum = candidate.split.left_sum;
  parent_sum.Add(candidate.split.right_sum);
  auto base_weight =
      evaluator.CalcWeight(candidate.nid, param, GradStats{parent_sum});

  auto left_weight = evaluator.CalcWeight(candidate.nid, param,
                                          GradStats{candidate.split.left_sum}) *
                     param.learning_rate;
  auto right_weight =
      evaluator.CalcWeight(candidate.nid, param,
                           GradStats{candidate.split.right_sum}) *
      param.learning_rate;

  tree.ExpandNode(
      candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
      candidate.split.DefaultLeft(), base_weight, left_weight, right_weight,
      candidate.split.loss_chg, parent_sum.GetHess(),
      candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());

  // Set up child constraints
  auto left_child = tree[candidate.nid].LeftChild();
  auto right_child = tree[candidate.nid].RightChild();
  tree_evaluator->AddSplit(candidate.nid, left_child, right_child,
                           tree[candidate.nid].SplitIndex(), left_weight,
                           right_weight);

  auto max_node = std::max(left_child, tree[candidate.nid].RightChild());
  max_node = std::max(candidate.nid, max_node);
  snode.resize(tree.GetNodes().size());
  snode.at(left_child).stats = candidate.split.left_sum;
  snode.at(left_child).root_gain = evaluator.CalcGain(
      candidate.nid, param, GradStats{candidate.split.left_sum});
  snode.at(right_child).stats = candidate.split.right_sum;
  snode.at(right_child).root_gain = evaluator.CalcGain(
      candidate.nid, param, GradStats{candidate.split.right_sum});

  interaction_constraints.Split(candidate.nid, tree[candidate.nid].SplitIndex(),
                                left_child, right_child);
}
} // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_APPROX_H_
