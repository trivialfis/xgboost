/**
 * Copyright 2025-2026, XGBoost contributors
 */
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator
#include <thrust/logical.h>                      // for any_of
#include <thrust/reduce.h>                       // for reduce_by_key
#include <thrust/sort.h>                         // for stable_sort_by_key

#include <algorithm>       // for max
#include <cuda/std/tuple>  // for get, make_tuple, tuple
#include <memory>          // for make_unique

#include "../../common/cuda_compat.cuh"  // for CUDA compatibility
#include "../../common/cuda_context.cuh"
#include "../../common/nvtx_utils.h"  // for xgboost_NVTX_FN_RANGE
#include "../tree_view.h"             // for MultiTargetTreeView
#include "evaluate_splits_impl.cuh"
#include "multi_evaluate_splits.cuh"  // for MultiEvalauteSplitInputs, MultiEvaluateSplitSharedInputs
#include "quantiser.cuh"              // for GradientQuantiser
#include "xgboost/base.h"             // for GradientPairInt64
#include "xgboost/span.h"             // for Span

namespace xgboost::tree::cuda_impl {
void MultiHistEvaluator::Reset(Context const *ctx,
                               common::Span<std::uint32_t const> feature_segments,
                               common::Span<FeatureType const> feature_types,
                               TrainParam const &param, bst_target_t n_targets) {
  CHECK_GT(feature_segments.size(), 0);
  CHECK_GT(n_targets, 0);
  auto n_features = static_cast<bst_feature_t>(feature_segments.size() - 1);
  this->tree_evaluator_ =
      std::make_unique<TreeEvaluator>(param, n_features, ctx->Device(), n_targets);
  this->need_sort_histogram_ = false;
  if (feature_types.empty()) {
    return;
  }

  auto feature_it = dh::make_counting_iterator<bst_feature_t>(0);
  auto max_cat_to_onehot = param.max_cat_to_onehot;
  this->need_sort_histogram_ =
      thrust::any_of(ctx->CUDACtx()->CTP(), feature_it, feature_it + n_features,
                     [=] XGBOOST_DEVICE(bst_feature_t fidx) {
                       if (!common::IsCat(feature_types, fidx)) {
                         return false;
                       }
                       auto n_cats = feature_segments[fidx + 1] - feature_segments[fidx];
                       return !common::UseOneHot(n_cats, max_cat_to_onehot);
                     });
}

[[nodiscard]] MultiExpandEntry MultiHistEvaluator::EvaluateSingleSplit(
    Context const *ctx, MultiEvaluateSplitInputs const &input,
    MultiEvaluateSplitSharedInputs const &shared_inputs) {
  dh::device_vector<MultiEvaluateSplitInputs> inputs{input};
  dh::device_vector<MultiExpandEntry> outputs(1);

  auto d_outputs = dh::ToSpan(outputs);
  this->EvaluateSplits(ctx, dh::ToSpan(inputs), shared_inputs, input.nidx, d_outputs);

  return outputs[0];
}

namespace {
// Sort histogram based on projected score, see CPU implementation for details.
void SortHistogram(Context const *ctx, MultiEvaluateSplitSharedInputs const &shared_inputs,
                   common::Span<MultiEvaluateSplitInputs const> d_inputs,
                   TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
                   dh::device_vector<std::size_t> *p_sorted_idx) {
  auto &sorted_idx = *p_sorted_idx;
  auto n_nodes = d_inputs.size();

  auto bins_per_tar = shared_inputs.n_total_bins_per_tar;
  std::size_t total_bins = static_cast<std::size_t>(n_nodes) * bins_per_tar;

  sorted_idx.resize(total_bins);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  using SortKey = cuda::std::tuple<std::size_t, bst_feature_t, double>;
  dh::device_vector<SortKey> keys(total_bins);

  auto cnt_it = dh::make_counting_iterator(0ul);
  thrust::transform(
      ctx->CUDACtx()->CTP(), cnt_it, cnt_it + total_bins, keys.begin(),
      [=] XGBOOST_DEVICE(std::size_t i) {
        d_sorted_idx[i] = i % bins_per_tar;  // segmented iota
        auto [nidx_in_set, bin_idx] = linalg::UnravelIndex(i, n_nodes, bins_per_tar);
        auto fidx =
            static_cast<bst_feature_t>(dh::SegmentId(shared_inputs.feature_segments, bin_idx));
        auto n_cats =
            shared_inputs.feature_segments[fidx + 1] - shared_inputs.feature_segments[fidx];
        bool is_partition = shared_inputs.IsCategorical(fidx) &&
                            !common::UseOneHot(n_cats, shared_inputs.param.max_cat_to_onehot);

        double sc = 0.0;
        if (!is_partition) {
          return cuda::std::make_tuple(nidx_in_set, fidx, sc);
        }

        auto const &node = d_inputs[nidx_in_set];
        for (bst_target_t t = 0; t < shared_inputs.Targets(); ++t) {
          auto quantizer = shared_inputs.roundings[t];
          auto target_hist = node.histogram.subspan(t * bins_per_tar, bins_per_tar);
          auto child_sum = quantizer.ToFloatingPoint(target_hist[bin_idx]);
          auto parent_sum = quantizer.ToFloatingPoint(node.parent_sum[t]);
          auto child_weight = evaluator.CalcWeightCat(child_sum);
          auto parent_weight = evaluator.CalcWeightCat(parent_sum);
          sc += child_weight * parent_weight;
        }
        return cuda::std::make_tuple(nidx_in_set, fidx, sc);
      });
  thrust::stable_sort_by_key(ctx->CUDACtx()->CTP(), keys.begin(), keys.end(), sorted_idx.begin(),
                             [] XGBOOST_DEVICE(SortKey const &l, SortKey const &r) {
                               // nidx_in_set
                               if (cuda::std::get<0>(l) != cuda::std::get<0>(r)) {
                                 return cuda::std::get<0>(l) < cuda::std::get<0>(r);
                               }
                               // fidx
                               if (cuda::std::get<1>(l) != cuda::std::get<1>(r)) {
                                 return cuda::std::get<1>(l) < cuda::std::get<1>(r);
                               }
                               // score
                               return cuda::std::get<2>(l) < cuda::std::get<2>(r);
                             });
}
}  // namespace

void MultiHistEvaluator::EvaluateSplits(Context const *ctx,
                                        common::Span<MultiEvaluateSplitInputs const> d_inputs,
                                        MultiEvaluateSplitSharedInputs const &shared_inputs,
                                        bst_node_t max_nidx,
                                        common::Span<MultiExpandEntry> out_splits) {
  auto n_targets = shared_inputs.Targets();
  auto evaluator = this->GetEvaluator();
  CHECK_GT(n_targets, 0);
  CHECK_EQ(n_targets, evaluator.n_targets);
  CHECK_EQ(n_targets, shared_inputs.roundings.size());
  CHECK_GE(shared_inputs.n_total_bins_per_tar, 1);
  auto n_features = shared_inputs.max_active_feature;
  CHECK_GE(n_features, 1);
  CHECK_LT(n_features, shared_inputs.feature_segments.size());

  std::uint32_t n_nodes = d_inputs.size();
  CHECK_EQ(n_nodes, out_splits.size());

  if (n_nodes == 0) {
    return;
  }

  // Allocate weight and split sum storage on demand for the maximum node ID being evaluated.
  this->AllocNodeWeight(max_nidx, n_targets);
  this->split_sums_.Alloc(max_nidx, n_targets);

  if (shared_inputs.cat_storage_size > 0) {
    this->AllocNodeCats(max_nidx, shared_inputs.cat_storage_size);
  }

  // The values are node-local bin indices. Sort by (node, feature, score)
  dh::device_vector<std::size_t> sorted_idx;
  if (this->need_sort_histogram_) {
    SortHistogram(ctx, shared_inputs, d_inputs, evaluator, &sorted_idx);
  }
  auto d_sorted_idx = common::Span<std::size_t const>{dh::ToSpan(sorted_idx)};

  dh::CachingDeviceUVector<MultiSplitCandidate> d_splits(n_nodes * n_features);
  if (n_targets <= 4 && shared_inputs.cat_storage_size == 0) {
    // Only the winning sums for each feature need to survive the fused kernel.
    scan_buffer_.resize(static_cast<std::size_t>(n_nodes) * n_features * n_targets);
    split_kernel::DispatchFusedTargets(n_targets, [&](auto targets) {
      dh::LaunchKernel{n_nodes * n_features, 32, 0, ctx->CUDACtx()->Stream()}(
          split_kernel::FusedEvaluateSplitsKernel<decltype(targets)::value,
                                                 MultiEvaluateSplitInputs,
                                                 MultiEvaluateSplitSharedInputs,
                                                 MultiSplitCandidate>,
          d_inputs, shared_inputs, n_features, evaluator, dh::ToSpan(scan_buffer_),
          dh::ToSpan(d_splits));
    });
  } else {
    // Calculate total scan buffer size needed for all nodes
    auto node_hist_size = shared_inputs.n_total_bins_per_tar * n_targets;
    std::size_t total_hist_size = node_hist_size * n_nodes;

    // Scan the histograms. One for forward and the other for backward.
    // Since there's only store op on the scan buffer, no need to initialize it.
    this->scan_buffer_.resize(total_hist_size * 2);

    // Create spans for each node's scan results
    auto in_scans = dh::ToSpan(this->scan_buffer_);
    auto d_scans = common::IterSpan{
        thrust::make_transform_iterator(
            dh::make_counting_iterator(0ul),
            [=] XGBOOST_DEVICE(std::size_t nidx_in_set) -> common::Span<GradientPairInt64> {
              return in_scans.subspan(nidx_in_set * node_hist_size * 2, node_hist_size * 2);
            }),
        n_nodes};
    // Launch histogram scan kernel, each warp handles one target of one feature of one node.
    {
      std::uint32_t constexpr kBlockThreads = 128;
      constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
      auto n_warps = n_nodes * n_targets * n_features;
      auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
      dh::LaunchKernel{n_blocks, kBlockThreads, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
          split_kernel::ScanHistogramKernel<kBlockThreads, MultiEvaluateSplitInputs,
                                            MultiEvaluateSplitSharedInputs, decltype(d_sorted_idx),
                                            decltype(d_scans)>,
          d_inputs, shared_inputs, n_features, d_sorted_idx, d_scans);
    }

    // Launch split evaluation kernel
    {
      std::uint32_t constexpr kBlockThreads = 32;
      constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
      auto n_warps = n_nodes * n_features;
      auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
      dh::LaunchKernel{n_blocks, kBlockThreads, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
          split_kernel::EvaluateSplitsKernel<kBlockThreads, MultiEvaluateSplitInputs,
                                             MultiEvaluateSplitSharedInputs, decltype(d_scans),
                                             MultiSplitCandidate>,
          d_inputs, shared_inputs, n_features, d_scans, evaluator, dh::ToSpan(d_splits));
    }

  }

  // Find best split for each node
  auto d_weights = this->GetNodeWeights(n_targets);
  auto d_split_sums = this->split_sums_.View();
  auto d_split_cats = dh::ToSpan(this->split_cats_);
  auto node_cat_storage_size = this->node_cat_storage_size_;
  auto s_d_splits = dh::ToSpan(d_splits);

  // Process results for each node
  // Find best splits among all features for all nodes
  auto key_it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
    // Returns nidx_in_set
    return i / n_features;
  });
  dh::DeviceUVector<MultiSplitCandidate> best_splits(out_splits.size(), MultiSplitCandidate{},
                                                     ctx->CUDACtx()->Stream());

  thrust::reduce_by_key(
      ctx->CUDACtx()->CTP(), key_it, key_it + s_d_splits.size(), dh::tcbegin(s_d_splits),
      thrust::make_discard_iterator(), best_splits.begin(), std::equal_to{},
      [=] XGBOOST_DEVICE(MultiSplitCandidate const &lhs, MultiSplitCandidate const &rhs) {
        if (lhs.loss_chg > rhs.loss_chg) {
          return lhs;
        }
        if (rhs.loss_chg > lhs.loss_chg) {
          return rhs;
        }
        return lhs.findex <= rhs.findex ? lhs : rhs;
      });
  auto d_best_splits = dh::ToSpan(best_splits);

  dh::LaunchN(n_nodes, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t nidx_in_set) {
    auto input = d_inputs[nidx_in_set];
    MultiSplitCandidate best_split = d_best_splits[nidx_in_set];
    common::Span<common::CatBitField::value_type> node_cats;
    if (!d_split_cats.empty()) {
      node_cats = d_split_cats.subspan(input.nidx * node_cat_storage_size, node_cat_storage_size);
      // Clear the bits, we use it to store the best candidate.
      for (auto &node_cat : node_cats) {
        node_cat = 0;
      }
    }

    // The root weight is required even when no valid split is found. In particular,
    // max_cat_threshold=1 does not enumerate any partition candidate.
    bst_node_t nidx = input.nidx;
    auto base_weight = d_weights.Base(nidx);
    auto roundings = shared_inputs.roundings;
    split_kernel::QuantizedGradientSum parent_sum{input.parent_sum.data(), roundings};
    double parent_gain = evaluator.CalcGain(nidx, shared_inputs.param, parent_sum);
    double parent_hess = 0;
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto g = roundings[t].ToFloatingPoint(input.parent_sum[t]);
      base_weight[t] = evaluator.CalcWeight(nidx, t, shared_inputs.param, g);
      parent_hess += g.GetHess();
    }

    if (best_split.child_sum.empty()) {
      // Invalid split
      out_splits[nidx_in_set] = {nidx, input.depth, best_split};
      out_splits[nidx_in_set].UpdateHessian(parent_hess, 0.0);
      return;
    }

    if (best_split.is_cat) {
      common::CatBitField cats{node_cats};
      if (!isnan(best_split.fvalue)) {  // OHE
        cats.Set(common::AsCat(best_split.fvalue));
      } else {  // Partition
        auto fidx = best_split.findex;
        auto f_begin = shared_inputs.feature_segments[fidx];
        auto n_bins =
            shared_inputs.feature_segments[fidx + 1] - shared_inputs.feature_segments[fidx];
        auto node_sorted_idx = d_sorted_idx.subspan(
            nidx_in_set * shared_inputs.n_total_bins_per_tar, shared_inputs.n_total_bins_per_tar);
        auto f_sorted_idx = node_sorted_idx.subspan(f_begin, n_bins);
        bst_bin_t partition = best_split.dir == kLeftDir
                                  ? static_cast<bst_bin_t>(best_split.thresh + 1)
                                  : static_cast<bst_bin_t>(best_split.thresh);
        KERNEL_CHECK(partition > 0);
        for (bst_bin_t i = 0; i < partition; ++i) {
          auto cat = shared_inputs.feature_values[f_sorted_idx[i]];
          cats.Set(common::AsCat(cat));
        }
      }
    }

    // Calculate weights for this node using the actual node id for persistent storage
    auto left_weight = d_weights.Left(nidx);
    auto right_weight = d_weights.Right(nidx);

    auto split_sum = best_split.child_sum;

    // Copy split sum to persistent buffer for loss-guide grow policy support.
    // The child_sum span in best_split points to scan_buffer_ which gets reused,
    // so we store it persistently indexed by node id.
    auto split_sum_dest = GetNodeSumImpl(d_split_sums, nidx, n_targets);
    split_kernel::QuantizedGradientSum child{split_sum.data(), roundings};
    split_kernel::QuantizedGradientDifference sibling{input.parent_sum.data(), split_sum.data(),
                                                      roundings};

    double left_hess = 0, right_hess = 0;  // Sum of child hessians across all targets

    for (bst_target_t t = 0; t < n_targets; ++t) {
      split_sum_dest[t] = split_sum[t];

      GradientPairPrecise lg, rg;
      if (best_split.dir == kRightDir) {
        // forward pass, split_sum is the left sum
        lg = child(t);
        rg = sibling(t);
      } else {
        // backward pass, split_sum is the right sum
        lg = sibling(t);
        rg = child(t);
      }
      auto [lw, rw] =
          evaluator.CalcSplitWeights(shared_inputs.param, nidx, best_split.findex, t, lg, rg);
      left_weight[t] = lw;
      right_weight[t] = rw;

      left_hess += lg.GetHess();
      right_hess += rg.GetHess();
    }

    // Set up the output entry after persisting its weights and split sum by node ID.
    out_splits[nidx_in_set] = {nidx, input.depth, best_split};
    out_splits[nidx_in_set].split.loss_chg -= parent_gain;
    out_splits[nidx_in_set].UpdateHessian(left_hess, right_hess);
  });
}

void MultiHistEvaluator::ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                                        common::Span<MultiExpandEntry const> h_candidates,
                                        common::Span<MultiExpandEntry const> d_candidates,
                                        bst_target_t n_targets) {
  xgboost_NVTX_FN_RANGE();
  CHECK_EQ(h_candidates.size(), d_candidates.size());
  CHECK(!h_candidates.empty());
  CHECK_EQ(n_targets, this->GetEvaluator().n_targets);

  auto weights = this->GetNodeWeights(n_targets);
  auto mt_tree = MultiTargetTreeView{ctx->Device(), false, p_tree};
  bst_node_t max_node{0};
  for (auto const &candidate : h_candidates) {
    max_node =
        std::max({max_node, p_tree->LeftChild(candidate.nidx), p_tree->RightChild(candidate.nidx)});
  }
  this->tree_evaluator_->AddSplits(
      d_candidates.size(), max_node, [=] XGBOOST_DEVICE(std::size_t i, bst_target_t t) {
        auto const &candidate = d_candidates[i];
        auto left_weight = weights.Left(candidate.nidx)[t];
        auto right_weight = weights.Right(candidate.nidx)[t];
        return TreeEvaluator::SplitBounds{candidate.nidx,
                                          mt_tree.LeftChild(candidate.nidx),
                                          mt_tree.RightChild(candidate.nidx),
                                          static_cast<bst_feature_t>(candidate.split.findex),
                                          left_weight,
                                          right_weight};
      });

  // Assign the node sums here, for the next evaluate split call.
  this->AllocNodeSum(max_node, n_targets);

  auto node_sums = this->node_sums_.View();
  // Use the internal split sums buffer instead of candidate.split.child_sum . It may be
  // stale in loss-guide grow policy (entries can remain in priority queue across
  // evaluation rounds).
  auto split_sums = this->split_sums_.View();

  dh::LaunchN(n_targets * d_candidates.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) {
                auto get_node_sum = [&](bst_node_t nidx) {
                  return GetNodeSumImpl(node_sums, nidx, n_targets);
                };
                auto nidx_in_set = i / n_targets;
                auto t = i % n_targets;

                auto const &candidate = d_candidates[nidx_in_set];
                auto const &best_split = candidate.split;

                auto parent_sum = get_node_sum(candidate.nidx);
                // Look up split sum from persistent buffer by node id.
                // Use split_targets for indexing since that's what was used during storage.
                auto split_sum = GetNodeSumImpl(split_sums, candidate.nidx, n_targets);
                auto left_sum = get_node_sum(mt_tree.LeftChild(candidate.nidx));
                auto right_sum = get_node_sum(mt_tree.RightChild(candidate.nidx));

                auto split_sum_t = split_sum[t];
                auto sibling_sum = parent_sum[t] - split_sum_t;
                if (best_split.dir == kRightDir) {
                  // forward pass, node_sum is the left sum
                  left_sum[t] = split_sum_t;
                  right_sum[t] = sibling_sum;
                } else {
                  // backward pass, node_sum is the right sum
                  right_sum[t] = split_sum_t;
                  left_sum[t] = sibling_sum;
                }
              });
}
}  // namespace xgboost::tree::cuda_impl
