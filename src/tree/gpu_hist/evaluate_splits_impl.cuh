/**
 * Copyright 2026, XGBoost Contributors
 * @brief Histogram scan and split evaluation shared by scalar and vector GPU trees.
 */
#pragma once

#include <cub/util_type.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda/ptx>
#include <cuda/std/functional>
#include <limits>
#include <type_traits>

#include "evaluate_splits.cuh"

namespace xgboost::tree::cuda_impl {
namespace split_kernel {

// Adapt scalar storage inside the kernel, avoiding input-conversion kernels and uploads.
XGBOOST_DEVICE inline MultiEvaluateSplitInputs GetInput(EvaluateSplitInputs const &input) {
  return {
      input.nidx, input.depth, {&input.parent_sum, 1}, input.feature_set, input.gradient_histogram};
}
XGBOOST_DEVICE inline MultiEvaluateSplitInputs GetInput(MultiEvaluateSplitInputs const &input) {
  return input;
}
XGBOOST_DEVICE inline MultiEvaluateSplitSharedInputs GetSharedInput(
    EvaluateSplitSharedInputs const &input, bst_feature_t max_active_features) {
  MultiEvaluateSplitSharedInputs result{};
  result.roundings = {&input.rounding, 1};
  result.feature_segments = input.feature_segments;
  result.feature_values = input.feature_values.data();
  result.feature_types = input.feature_types;
  result.n_total_bins_per_tar = input.feature_values.size();
  result.max_active_feature = max_active_features;
  result.param = input.param;
  result.is_dense = input.is_dense;
  return result;
}
XGBOOST_DEVICE inline MultiEvaluateSplitSharedInputs GetSharedInput(
    MultiEvaluateSplitSharedInputs const &input, bst_feature_t) {
  return input;
}

XGBOOST_DEVICE inline void StoreCandidate(MultiSplitCandidate const &split,
                                          MultiEvaluateSplitInputs const &,
                                          MultiSplitCandidate *out) {
  *out = split;
}
XGBOOST_DEVICE inline void StoreCandidate(MultiSplitCandidate const &split,
                                          MultiEvaluateSplitInputs const &input,
                                          DeviceSplitCandidate *out) {
  *out = DeviceSplitCandidate{};
  if (split.child_sum.empty()) {
    return;
  }
  auto child = split.child_sum[0];
  auto sibling = input.parent_sum[0] - child;
  auto left = split.dir == kRightDir ? child : sibling;
  auto right = split.dir == kRightDir ? sibling : child;
  if (split.is_cat) {
    auto thresh = isnan(split.fvalue) ? split.thresh : static_cast<bst_cat_t>(split.fvalue);
    out->UpdateCat(split.loss_chg, split.dir, thresh, split.findex, left, right);
  } else {
    out->Update(split.loss_chg, split.dir, split.fvalue, split.findex, left, right, false);
  }
}

/**
 * @brief Calculate the gradient index for the reverse pass
 *
 * @note All inputs are global across features.
 */
XGBOOST_DEV_INLINE bst_bin_t RevBinIdx(bst_bin_t gidx_begin, bst_bin_t gidx_end,
                                       bst_bin_t bin_idx) {
  return gidx_begin + (gidx_end - bin_idx - 1);
}

// Scan the histogram in 2 dim for all nodes
struct ScanHistogramAgent {
  using WarpScanT = cub::WarpScan<GradientPairInt64>;

  typename WarpScanT::TempStorage *tmp_storage;
  bst_bin_t gidx_begin;
  bst_bin_t gidx_end;
  bst_target_t n_targets;

  template <typename BinIndexFn>
  __device__ void ScanFeature(GradientPairInt64 const *node_histogram,
                              GradientPairInt64 *scan_result, bst_target_t t,
                              BinIndexFn &&bin_idx_fn) const {
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
    // The forward pass and the backward pass differs in where the bin is read, which is
    // specified by the callback bin_idx_fn(). They write to the same output location.
    GradientPairInt64 warp_aggregate;
    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;
      // Read from histogram: [target][bins]
      auto bin = thread_active ? node_histogram[bin_idx_fn(bin_idx)] : GradientPairInt64{};
      if (lane_id == 0) {
        bin += warp_aggregate;
      }
      WarpScanT(*tmp_storage).InclusiveScan(bin, bin, cuda::std::plus{}, warp_aggregate);
      // Required by the warp scan.
      __syncwarp();
      if (thread_active) {
        // Write to scan result: [bins][targets]
        // The layout is changed from target-major to bin-major here.
        scan_result[bin_idx * n_targets + t] = bin;
      }
    }
  }
  // Forward scan pass
  __device__ void Forward(GradientPairInt64 const *node_histogram,
                          common::Span<GradientPairInt64> scan_result, bst_target_t t) const {
    this->ScanFeature(node_histogram, scan_result.data(), t, cuda::std::identity{});
  }
  // Backward scan pass for missing values
  __device__ void Backward(GradientPairInt64 const *node_histogram,
                           common::Span<GradientPairInt64> scan_result, bst_target_t t) const {
    this->ScanFeature(node_histogram, scan_result.data(), t,
                      [&](bst_bin_t bin_idx) { return RevBinIdx(gidx_begin, gidx_end, bin_idx); });
  }

  // One-hot pass for categorical features.
  //
  // For categorical features the two scan-buffer regions are not a forward/backward scan
  // duality; they hold the two independent missing-direction candidates. We write the
  // *non-missing child sum* for each direction:
  //   - region_others: the matching category goes right with missing (missing-right). The
  //     written non-missing child is the left.
  //   - region_match: the matching category goes right without missing (missing-left). The
  //     written non-missing child is the right.
  __device__ void OneHot(GradientPairInt64 const *node_histogram,
                         common::Span<GradientPairInt64> region_others,
                         common::Span<GradientPairInt64> region_match, bst_target_t t) const {
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
    // Feature sum across all bins for this target.
    GradientPairInt64 local{};
    for (auto bin_idx = gidx_begin + lane_id; bin_idx < gidx_end; bin_idx += dh::WarpThreads()) {
      local += node_histogram[bin_idx];
    }
    auto feature_sum = WarpSum(local);
    // Per-bin child sums, written in the bin-major layout: [bins][targets].
    for (auto bin_idx = gidx_begin + lane_id; bin_idx < gidx_end; bin_idx += dh::WarpThreads()) {
      auto bin = node_histogram[bin_idx];
      region_others[bin_idx * n_targets + t] = feature_sum - bin;
      region_match[bin_idx * n_targets + t] = bin;
    }
  }

  template <typename SortedIndex>
  __device__ void Partition(GradientPairInt64 const *node_histogram, SortedIndex sorted_idx,
                            common::Span<GradientPairInt64> forward,
                            common::Span<GradientPairInt64> backward, bst_target_t t) const {
    this->ScanFeature(node_histogram, forward.data(), t,
                      [&](bst_bin_t bin_idx) { return sorted_idx[bin_idx]; });
    this->ScanFeature(node_histogram, backward.data(), t, [&](bst_bin_t bin_idx) {
      return sorted_idx[RevBinIdx(gidx_begin, gidx_end, bin_idx)];
    });
  }
};

// The scan kernel reads from target-major histogram layout and writes the bin-major scan
// buffer. This helps us keep a reference to the bin in the split candidate.
template <std::int32_t kBlockThreads, typename Input, typename SharedInput, typename SortedIndex,
          typename OutputBinScans>
__global__ __launch_bounds__(kBlockThreads) void ScanHistogramKernel(
    common::Span<Input const> nodes, SharedInput shared_input, bst_feature_t max_active_features,
    SortedIndex sorted_idx, OutputBinScans outputs) {
  static_assert(kBlockThreads % dh::WarpThreads() == 0);
  auto shared = GetSharedInput(shared_input, max_active_features);

  constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
  auto const warp_id_in_blk = static_cast<std::int32_t>(threadIdx.x) / dh::WarpThreads();
  // The warp index across the entire grid
  auto const warp_id = warp_id_in_blk + kWarpsPerBlk * blockIdx.x;
  bst_target_t const n_targets = shared.Targets();
  auto const n_valid_warps = nodes.size() * shared.max_active_feature * n_targets;

  if (warp_id >= n_valid_warps) {
    return;
  }

  auto [nidx_in_set, fidx_in_set, target_idx] =
      linalg::UnravelIndex(warp_id, nodes.size(), shared.max_active_feature, n_targets);
  auto node = GetInput(nodes[nidx_in_set]);
  auto out = outputs[nidx_in_set];
  // This node might have a smaller number of sampled features.
  if (fidx_in_set >= node.feature_set.size()) {
    return;
  }
  auto fidx = node.feature_set[fidx_in_set];
  // The histogram is full, regardless of whether a feature is sampled.
  bst_bin_t gidx_begin = shared.feature_segments[fidx];
  bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

  using AgentT = ScanHistogramAgent;
  __shared__ typename AgentT::WarpScanT::TempStorage tmp_storage[kWarpsPerBlk];
  ScanHistogramAgent agent{&tmp_storage[warp_id_in_blk], gidx_begin, gidx_end, n_targets};
  auto t_hist = node.histogram.data() + shared.n_total_bins_per_tar * target_idx;
  // Scalar histogram spans can include extra storage; scan offsets use the cut layout.
  auto node_hist_size = shared.n_total_bins_per_tar * n_targets;

  if (shared.IsCategorical(fidx)) {
    auto first = out.subspan(0, node_hist_size);
    auto second = out.subspan(node_hist_size, node_hist_size);
    auto n_bins = gidx_end - gidx_begin;
    if (common::UseOneHot(n_bins, shared.param.max_cat_to_onehot)) {
      // Both regions are always required (independent missing-directions), so `one_pass`
      // does not apply to categorical features.
      agent.OneHot(t_hist, first, second, target_idx);
    } else {
      auto node_sorted_idx = sorted_idx.subspan(nidx_in_set * shared.n_total_bins_per_tar,
                                                shared.n_total_bins_per_tar);
      agent.Partition(t_hist, node_sorted_idx, first, second, target_idx);
    }
    return;
  }

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = out.subspan(0, node_hist_size);
    agent.Forward(t_hist, forward, target_idx);
  }
  if (shared.NeedBackward()) {
    auto backward = out.subspan(node_hist_size, node_hist_size);
    agent.Backward(t_hist, backward, target_idx);
  }
}

struct QuantizedGradientSum {
  GradientPairInt64 const *values;
  common::Span<GradientQuantiser const> roundings;

  [[nodiscard]] XGBOOST_DEVICE std::size_t Size() const { return roundings.size(); }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise operator()(std::size_t t) const {
    return roundings.data()[t].ToFloatingPoint(values[t]);
  }
};

struct QuantizedGradientDifference {
  GradientPairInt64 const *parent;
  GradientPairInt64 const *child;
  common::Span<GradientQuantiser const> roundings;

  [[nodiscard]] XGBOOST_DEVICE std::size_t Size() const { return roundings.size(); }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise operator()(std::size_t t) const {
    return roundings.data()[t].ToFloatingPoint(parent[t] - child[t]);
  }
};

static_assert(std::is_trivially_copyable_v<QuantizedGradientSum>);
static_assert(std::is_trivially_copyable_v<QuantizedGradientDifference>);

struct EvaluateSplitAgent {
  using ArgMaxT = cub::KeyValuePair<std::uint32_t, double>;
  using MaxReduceT = cub::WarpReduce<ArgMaxT>;

  typename MaxReduceT::TempStorage *temp_storage;
  bst_feature_t fidx;
  TreeEvaluator::SplitEvaluator<EvalParam> evaluator;

  // Calculate the split gain for one bin. `child_scan` has the bin-major layout
  // [bins][targets] and stores the non-missing child sum.
  template <DefaultDirection d_dir>
  static __device__ double ComputeGain(MultiEvaluateSplitInputs const &node,
                                       MultiEvaluateSplitSharedInputs const &shared,
                                       common::Span<GradientPairInt64 const> child_scan,
                                       bst_bin_t bin_idx, bst_target_t n_targets,
                                       bst_feature_t fidx,
                                       TreeEvaluator::SplitEvaluator<EvalParam> const &evaluator) {
    auto offset = bin_idx * n_targets;
    auto child_values = child_scan.subspan(offset, n_targets);
    QuantizedGradientSum child{child_values.data(), shared.roundings};
    QuantizedGradientDifference sibling{node.parent_sum.data(), child_values.data(),
                                        shared.roundings};
    if constexpr (d_dir == kRightDir) {
      return evaluator.CalcSplitGain(shared.param, node.nidx, fidx, child, sibling);
    } else {
      return evaluator.CalcSplitGain(shared.param, node.nidx, fidx, sibling, child);
    }
  }

  template <DefaultDirection d_dir>
  __device__ void Numerical(MultiEvaluateSplitInputs const &node,
                            MultiEvaluateSplitSharedInputs const &shared,
                            common::Span<GradientPairInt64 const> node_scan,
                            MultiSplitCandidate *best_split) {
    // Calculate split gain for each bin
    auto n_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, node_scan, bin_idx, n_targets,
                                                       fidx, evaluator)
                                  : kNullGain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        // Update
        bst_bin_t split_gidx = bin_idx;
        if (d_dir == kLeftDir) {
          split_gidx = RevBinIdx(gidx_begin, gidx_end, bin_idx);
        }
        float fvalue;
        if (d_dir == kRightDir) {
          fvalue = shared.feature_values[split_gidx];
        } else {
          if (split_gidx == gidx_begin) {
            fvalue = -std::numeric_limits<float>::infinity();
          } else {
            fvalue = shared.feature_values[split_gidx - 1];
          }
        }
        // Scan result layout: [bins][targets] - all targets for this bin are contiguous
        // bin_idx is the global bin index
        auto scan_bin_offset = bin_idx * n_targets;
        auto scan_bin = node_scan.subspan(scan_bin_offset, n_targets);
        // Missing values go to right in the forward pass, go to left in the backward pass.
        best_split->Update(gain, d_dir, fvalue, fidx, scan_bin, false);
      }

      __syncwarp();
    }
  }

  // One-hot split for a categorical feature. `region` holds the non-missing child sum for
  // one missing-direction (see ScanHistogramAgent::OneHot). Unlike the numerical backward
  // pass there is no reverse indexing, and the split value is the category id stored at the
  // bin. `d_dir` is the default (missing) direction: kRightDir pairs with `region_others`,
  // kLeftDir pairs with `region_match`.
  template <DefaultDirection d_dir>
  __device__ void OneHot(MultiEvaluateSplitInputs const &node,
                         MultiEvaluateSplitSharedInputs const &shared,
                         common::Span<GradientPairInt64 const> region,
                         MultiSplitCandidate *best_split) {
    auto n_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, region, bin_idx, n_targets,
                                                       fidx, evaluator)
                                  : kNullGain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        // The split value is the category id (the cut value at this bin).
        float fvalue = shared.feature_values[bin_idx];
        // The scan_bin is directionless, `d_dir` is the carrier of the missing
        // direction. We use it to recover the bin value and sibling value later.
        auto scan_bin = region.subspan(bin_idx * n_targets, n_targets);
        best_split->Update(gain, d_dir, fvalue, fidx, scan_bin, /*cat=*/true);
      }

      __syncwarp();
    }
  }

  template <DefaultDirection d_dir>
  __device__ void Partition(MultiEvaluateSplitInputs const &node,
                            MultiEvaluateSplitSharedInputs const &shared,
                            common::Span<GradientPairInt64 const> node_scan,
                            MultiSplitCandidate *best_split) {
    auto n_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];
    bst_bin_t n_bins_feature = gidx_end - gidx_begin;
    bst_bin_t n_bins = std::min(shared.param.max_cat_threshold, n_bins_feature);
    if (n_bins <= 1) {
      return;
    }

    // A partition must leave at least one category on each side.
    for (bst_bin_t scan_begin = gidx_begin; scan_begin < gidx_begin + n_bins - 1;
         scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_begin + n_bins - 1;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, node_scan, bin_idx, n_targets,
                                                       fidx, evaluator)
                                  : kNullGain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        auto scan_offset = bin_idx - gidx_begin;
        auto thresh = d_dir == kLeftDir ? scan_offset
                                        : static_cast<bst_bin_t>(n_bins_feature - scan_offset - 1);
        auto scan_bin = node_scan.subspan(bin_idx * n_targets, n_targets);
        // The forward scan selects a right-child prefix with missing values on the
        // left. The backward scan selects a left-child suffix with missing on the right.
        best_split->UpdateCat(gain, d_dir, static_cast<bst_cat_t>(thresh), fidx, scan_bin);
      }

      __syncwarp();
    }
  }
};

// Find the best split based on the scan result
//
// The scan buffer has a bin-major layout.
template <std::int32_t kBlockThreads, typename Input, typename SharedInput, typename BinScans,
          typename Output>
__global__ __launch_bounds__(kBlockThreads) void EvaluateSplitsKernel(
    common::Span<Input const> nodes, SharedInput shared_input, bst_feature_t max_active_features,
    BinScans bin_scans, TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
    common::Span<Output> out_candidates) {
  static_assert(kBlockThreads % dh::WarpThreads() == 0);
  auto shared = GetSharedInput(shared_input, max_active_features);

  constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
  auto const warp_id_in_blk = static_cast<std::int32_t>(threadIdx.x) / dh::WarpThreads();
  // The warp index across the entire grid
  auto const warp_id = warp_id_in_blk + kWarpsPerBlk * blockIdx.x;
  auto const n_valid_warps = nodes.size() * shared.max_active_feature;

  if (warp_id >= n_valid_warps) {
    return;
  }

  using AgentT = EvaluateSplitAgent;
  __shared__ typename AgentT::MaxReduceT::TempStorage temp_storage[kWarpsPerBlk];
  __shared__ cub::Uninitialized<MultiSplitCandidate> split_storage[kWarpsPerBlk];
  auto &best_split = split_storage[warp_id_in_blk].Alias();
  auto lane_id = cuda::ptx::get_sreg_laneid();
  if (lane_id == 0) {
    best_split = MultiSplitCandidate{};
  }
  __syncwarp();

  const auto nidx = warp_id / shared.max_active_feature;
  auto node = GetInput(nodes[nidx]);

  bst_feature_t fidx_in_set = warp_id - (nidx * shared.max_active_feature);
  auto candidate_idx = nidx * shared.max_active_feature + fidx_in_set;
  // Every feature slot must be written, including unused slots after feature sampling.
  if (fidx_in_set >= node.feature_set.size()) {
    if (lane_id == 0) {
      StoreCandidate(best_split, node, &out_candidates[candidate_idx]);
    }
    return;
  }
  auto fidx = node.feature_set[fidx_in_set];
  AgentT agent{&temp_storage[warp_id_in_blk], fidx, evaluator};
  auto node_hist_size = shared.n_total_bins_per_tar * shared.Targets();

  if (shared.IsCategorical(fidx)) {
    auto first = bin_scans[nidx].subspan(0, node_hist_size);
    auto second = bin_scans[nidx].subspan(node_hist_size, node_hist_size);
    auto n_bins = shared.feature_segments[fidx + 1] - shared.feature_segments[fidx];
    if (common::UseOneHot(n_bins, shared.param.max_cat_to_onehot)) {
      // Both regions are always evaluated (independent missing-directions), so `one_pass`
      // does not apply to categorical features.
      agent.template OneHot<kRightDir>(node, shared, first, &best_split);
      agent.template OneHot<kLeftDir>(node, shared, second, &best_split);
    } else {
      agent.template Partition<kLeftDir>(node, shared, first, &best_split);
      agent.template Partition<kRightDir>(node, shared, second, &best_split);
    }
  } else {
    if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
      auto forward = bin_scans[nidx].subspan(0, node_hist_size);
      agent.template Numerical<kRightDir>(node, shared, forward, &best_split);
    }
    if (shared.NeedBackward()) {
      auto backward = bin_scans[nidx].subspan(node_hist_size, node_hist_size);
      agent.template Numerical<kLeftDir>(node, shared, backward, &best_split);
    }
  }
  __syncwarp();
  if (lane_id == 0) {
    StoreCandidate(best_split, node, &out_candidates[candidate_idx]);
  }
}
}  // namespace split_kernel
}  // namespace xgboost::tree::cuda_impl
