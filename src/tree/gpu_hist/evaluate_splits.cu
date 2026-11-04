/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator
#include <thrust/logical.h>                      // for any_of
#include <thrust/reduce.h>                       // for reduce
#include <thrust/sort.h>                         // for stable_sort_by_key

#include <algorithm>                               // for max_element
#include <cub/device/device_segmented_reduce.cuh>  // for DeviceSegmentedReduce
#include <cub/util_type.cuh>                       // for KeyValuePair
#include <cub/warp/warp_reduce.cuh>                // for WarpReduce
#include <cub/warp/warp_scan.cuh>                  // for WarpScan
#include <cuda/ptx>                                // for get_sreg_laneid
#include <cuda/std/functional>                     // for identity, plus
#include <cuda/std/tuple>                          // for get, make_tuple, tuple
#include <limits>                                  // for numeric_limits
#include <memory>                                  // for make_unique
#include <type_traits>                             // for is_trivially_copyable_v

#include "../../common/cuda_compat.cuh"  // for CUDA compatibility
#include "../../common/cuda_context.cuh"
#include "../../common/hist_util.h"   // for HistogramCuts
#include "../../common/nvtx_utils.h"  // for xgboost_NVTX_FN_RANGE
#include "../tree_view.h"             // for MultiTargetTreeView
#include "evaluate_splits.cuh"        // for GPUHistEvaluator
#include "quantiser.cuh"              // for GradientQuantiser
#include "xgboost/base.h"             // for GradientPairInt64
#include "xgboost/span.h"             // for Span

namespace xgboost::tree::cuda_impl {
namespace {
/**
 * @brief Calculate the gradient index for the reverse pass
 *
 * @note All inputs are global across features.
 */
__device__ bst_bin_t RevBinIdx(bst_bin_t gidx_begin, bst_bin_t gidx_end, bst_bin_t bin_idx) {
  return gidx_begin + (gidx_end - bin_idx - 1);
}

// Scan the histogram in 2 dim for all nodes
struct ScanHistogramAgent {
  using WarpScanT = cub::WarpScan<GradientPairInt64>;

  typename WarpScanT::TempStorage *tmp_storage;
  bst_bin_t gidx_begin;
  bst_bin_t gidx_end;
  bst_target_t n_split_targets;

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
        scan_result[bin_idx * n_split_targets + t] = bin;
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
      region_others[bin_idx * n_split_targets + t] = feature_sum - bin;
      region_match[bin_idx * n_split_targets + t] = bin;
    }
  }

  __device__ void Partition(GradientPairInt64 const *node_histogram,
                            common::Span<std::size_t const> sorted_idx,
                            common::Span<GradientPairInt64> forward,
                            common::Span<GradientPairInt64> backward, bst_target_t t) const {
    this->ScanFeature(node_histogram, forward.data(), t,
                      [&](bst_bin_t bin_idx) { return sorted_idx[bin_idx]; });
    this->ScanFeature(node_histogram, backward.data(), t, [&](bst_bin_t bin_idx) {
      return sorted_idx[RevBinIdx(gidx_begin, gidx_end, bin_idx)];
    });
  }
};
}  // namespace

// The scan kernel reads from target-major histogram layout and writes the bin-major scan
// buffer. This helps us keep a reference to the bin in the split candidate.
template <std::int32_t kBlockThreads, typename OutputBinScans>
__global__ __launch_bounds__(kBlockThreads) void ScanHistogramKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
    common::Span<std::size_t const> sorted_idx, OutputBinScans outputs) {
  static_assert(kBlockThreads % dh::WarpThreads() == 0);

  constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
  auto const warp_id_in_blk = static_cast<std::int32_t>(threadIdx.x) / dh::WarpThreads();
  // The warp index across the entire grid
  auto const warp_id = warp_id_in_blk + kWarpsPerBlk * blockIdx.x;
  bst_target_t const n_split_targets = shared.Targets();
  auto const n_valid_warps = nodes.size() * shared.max_active_feature * n_split_targets;

  if (warp_id >= n_valid_warps) {
    return;
  }

  auto [nidx_in_set, fidx_in_set, target_idx] =
      linalg::UnravelIndex(warp_id, nodes.size(), shared.max_active_feature, n_split_targets);
  auto const &node = nodes[nidx_in_set];
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
  ScanHistogramAgent agent{&tmp_storage[warp_id_in_blk], gidx_begin, gidx_end, n_split_targets};
  auto t_hist =
      node.histogram.subspan(shared.n_total_bins_per_tar * target_idx, shared.n_total_bins_per_tar);

  if (shared.IsCategorical(fidx)) {
    auto first = out.subspan(0, node.histogram.size());
    auto second = out.subspan(node.histogram.size(), node.histogram.size());
    auto n_bins = gidx_end - gidx_begin;
    if (common::UseOneHot(n_bins, shared.param.max_cat_to_onehot)) {
      // Both regions are always required (independent missing-directions), so `one_pass`
      // does not apply to categorical features.
      agent.OneHot(t_hist.data(), first, second, target_idx);
    } else {
      auto node_sorted_idx = sorted_idx.subspan(nidx_in_set * shared.n_total_bins_per_tar,
                                                shared.n_total_bins_per_tar);
      agent.Partition(t_hist.data(), node_sorted_idx, first, second, target_idx);
    }
    return;
  }

  if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
    auto forward = out.subspan(0, node.histogram.size());
    agent.Forward(t_hist.data(), forward, target_idx);
  }
  if (shared.NeedBackward()) {
    auto backward = out.subspan(node.histogram.size(), node.histogram.size());
    agent.Backward(t_hist.data(), backward, target_idx);
  }
}

namespace {
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
                                       bst_bin_t bin_idx, bst_target_t n_split_targets,
                                       bst_feature_t fidx,
                                       TreeEvaluator::SplitEvaluator<EvalParam> const &evaluator) {
    auto offset = bin_idx * n_split_targets;
    auto child_values = child_scan.subspan(offset, n_split_targets);
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
    auto n_split_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, node_scan, bin_idx,
                                                       n_split_targets, fidx, evaluator)
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
        auto scan_bin_offset = bin_idx * n_split_targets;
        auto scan_bin = node_scan.subspan(scan_bin_offset, n_split_targets);
        // Missing values go to right in the forward pass, go to left in the backward pass.
        best_split->Update(gain, d_dir, fvalue, fidx, scan_bin, false);
      }

      __syncwarp();
    }
  }

  // One-hot split for a categorical feature. Both regions store a non-missing child sum
  // (see ScanHistogramAgent::OneHot). Compare missing directions for each category before
  // reducing across categories, as in the CPU evaluator.
  __device__ void OneHot(MultiEvaluateSplitInputs const &node,
                         MultiEvaluateSplitSharedInputs const &shared,
                         common::Span<GradientPairInt64 const> others,
                         common::Span<GradientPairInt64 const> match,
                         MultiSplitCandidate *best_split) {
    auto n_split_targets = shared.Targets();
    auto lane_id = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());

    bst_bin_t gidx_begin = shared.feature_segments[fidx];
    bst_bin_t gidx_end = shared.feature_segments[fidx + 1];

    for (auto scan_begin = gidx_begin; scan_begin < gidx_end; scan_begin += dh::WarpThreads()) {
      auto bin_idx = scan_begin + lane_id;
      bool thread_active = bin_idx < gidx_end;

      auto constexpr kNullGain = -std::numeric_limits<double>::infinity();
      auto left_gain = thread_active ? ComputeGain<kLeftDir>(node, shared, match, bin_idx,
                                                             n_split_targets, fidx, evaluator)
                                     : kNullGain;
      auto right_gain = thread_active ? ComputeGain<kRightDir>(node, shared, others, bin_idx,
                                                               n_split_targets, fidx, evaluator)
                                      : kNullGain;
      // Match CPU enumeration: missing-left wins a tie for the same category.
      auto dir = right_gain > left_gain ? kRightDir : kLeftDir;
      auto gain = dir == kLeftDir ? left_gain : right_gain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        // The split value is the category id (the cut value at this bin).
        float fvalue = shared.feature_values[bin_idx];
        auto region = dir == kLeftDir ? match : others;
        auto scan_bin = region.subspan(bin_idx * n_split_targets, n_split_targets);
        best_split->Update(gain, dir, fvalue, fidx, scan_bin, /*cat=*/true);
      }

      __syncwarp();
    }
  }

  template <DefaultDirection d_dir>
  __device__ void Partition(MultiEvaluateSplitInputs const &node,
                            MultiEvaluateSplitSharedInputs const &shared,
                            common::Span<GradientPairInt64 const> node_scan,
                            MultiSplitCandidate *best_split) {
    auto n_split_targets = shared.Targets();
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
      double gain = thread_active ? ComputeGain<d_dir>(node, shared, node_scan, bin_idx,
                                                       n_split_targets, fidx, evaluator)
                                  : kNullGain;

      auto best = MaxReduceT(*temp_storage).Reduce({threadIdx.x, gain}, cub::ArgMax{});
      auto best_thread = __shfl_sync(dh::WarpFullMask(), best.key, 0);

      if (threadIdx.x == best_thread && !isinf(gain)) {
        auto scan_offset = bin_idx - gidx_begin;
        auto thresh = d_dir == kLeftDir ? scan_offset
                                        : static_cast<bst_bin_t>(n_bins_feature - scan_offset - 1);
        auto scan_bin = node_scan.subspan(bin_idx * n_split_targets, n_split_targets);
        // The forward scan selects a right-child prefix with missing values on the
        // left. The backward scan selects a left-child suffix with missing on the right.
        best_split->UpdateCat(gain, d_dir, static_cast<bst_cat_t>(thresh), fidx, scan_bin);
      }

      __syncwarp();
    }
  }
};
}  // namespace

// Find the best split based on the scan result
//
// The scan buffer has a bin-major layout.
template <std::int32_t kBlockThreads, typename BinScans>
__global__ __launch_bounds__(kBlockThreads) void EvaluateSplitsKernel(
    common::Span<MultiEvaluateSplitInputs const> nodes, MultiEvaluateSplitSharedInputs shared,
    BinScans bin_scans, TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
    common::Span<MultiSplitCandidate> out_candidates) {
  static_assert(kBlockThreads % dh::WarpThreads() == 0);

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
  // Aligned && shared storage for best_split
  __shared__ cub::Uninitialized<MultiSplitCandidate> split_storage[kWarpsPerBlk];
  auto &best_split = split_storage[warp_id_in_blk].Alias();
  auto lane_id = cuda::ptx::get_sreg_laneid();
  if (lane_id == 0) {
    best_split = MultiSplitCandidate{};
  }
  __syncwarp();

  const auto nidx = warp_id / shared.max_active_feature;
  auto const &node = nodes[nidx];

  bst_feature_t fidx_in_set = warp_id - (nidx * shared.max_active_feature);
  // The number of candidates is allocated using active features.
  auto candidate_idx = nidx * shared.max_active_feature + fidx_in_set;
  // This node might have a smaller number of sampled features.
  if (fidx_in_set >= node.feature_set.size()) {
    if (lane_id == 0) {
      out_candidates[candidate_idx] = best_split;
    }
    return;
  }
  auto fidx = node.feature_set[fidx_in_set];
  AgentT agent{&temp_storage[warp_id_in_blk], fidx, evaluator};

  if (shared.IsCategorical(fidx)) {
    auto first = bin_scans[nidx].subspan(0, node.histogram.size());
    auto second = bin_scans[nidx].subspan(node.histogram.size(), node.histogram.size());
    auto n_bins = shared.feature_segments[fidx + 1] - shared.feature_segments[fidx];
    if (common::UseOneHot(n_bins, shared.param.max_cat_to_onehot)) {
      // Both regions are always evaluated (independent missing-directions), so `one_pass`
      // does not apply to categorical features.
      agent.OneHot(node, shared, first, second, &best_split);
    } else {
      agent.template Partition<kLeftDir>(node, shared, first, &best_split);
      agent.template Partition<kRightDir>(node, shared, second, &best_split);
    }
  } else {
    if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
      auto forward = bin_scans[nidx].subspan(0, node.histogram.size());
      agent.template Numerical<kRightDir>(node, shared, forward, &best_split);
    }
    if (shared.NeedBackward()) {
      auto backward = bin_scans[nidx].subspan(node.histogram.size(), node.histogram.size());
      agent.template Numerical<kLeftDir>(node, shared, backward, &best_split);
    }
  }
  __syncwarp();
  if (lane_id == 0) {
    out_candidates[candidate_idx] = best_split;
  }
}

void GPUHistEvaluator::Reset(Context const *ctx, common::Span<std::uint32_t const> feature_segments,
                             common::Span<FeatureType const> feature_types, TrainParam const &param,
                             bst_target_t n_split_targets) {
  CHECK_GT(feature_segments.size(), 0);
  CHECK_GT(n_split_targets, 0);
  copy_stream_.View().Sync();
  this->node_cat_storage_size_ = 0;
  this->split_cats_.clear();
  auto n_features = static_cast<bst_feature_t>(feature_segments.size() - 1);
  this->tree_evaluator_ =
      std::make_unique<TreeEvaluator>(param, n_features, ctx->Device(), n_split_targets);
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

[[nodiscard]] MultiExpandEntry GPUHistEvaluator::EvaluateSingleSplit(
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
                   dh::CachingDeviceUVector<std::size_t> *p_sorted_idx) {
  auto &sorted_idx = *p_sorted_idx;
  auto n_nodes = d_inputs.size();

  auto bins_per_tar = shared_inputs.n_total_bins_per_tar;
  std::size_t total_bins = static_cast<std::size_t>(n_nodes) * bins_per_tar;

  sorted_idx.resize(total_bins);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  using SortKey = cuda::std::tuple<std::size_t, bst_feature_t, double>;
  dh::caching_device_vector<SortKey> keys(total_bins);

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
          // Match CPU ordering for a single split target, including projected gradients.
          if (shared_inputs.Targets() == 1) {
            sc = child_weight;
            break;
          }
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

void GPUHistEvaluator::EvaluateSplits(Context const *ctx,
                                      common::Span<MultiEvaluateSplitInputs const> d_inputs,
                                      MultiEvaluateSplitSharedInputs const &shared_inputs,
                                      bst_node_t max_nidx,
                                      common::Span<MultiExpandEntry> out_splits) {
  auto n_split_targets = shared_inputs.Targets();
  auto evaluator = this->GetEvaluator();
  CHECK_GT(n_split_targets, 0);
  CHECK_EQ(n_split_targets, evaluator.n_targets);
  CHECK_EQ(n_split_targets, shared_inputs.roundings.size());
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
  this->AllocNodeWeight(max_nidx, n_split_targets);
  this->split_sums_.Alloc(max_nidx, n_split_targets);

  // Calculate total scan buffer size needed for all nodes
  auto node_hist_size = shared_inputs.n_total_bins_per_tar * n_split_targets;
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
  if (shared_inputs.cat_storage_size > 0) {
    this->AllocNodeCats(max_nidx, shared_inputs.cat_storage_size);
  }

  // The values are node-local bin indices. Sort by (node, feature, score)
  dh::CachingDeviceUVector<std::size_t> sorted_idx;
  if (this->need_sort_histogram_) {
    SortHistogram(ctx, shared_inputs, d_inputs, evaluator, &sorted_idx);
  }
  auto d_sorted_idx = common::Span<std::size_t const>{dh::ToSpan(sorted_idx)};

  // Launch histogram scan kernel, each warp handles one target of one feature of one node.
  {
    std::uint32_t constexpr kBlockThreads = 128;
    constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
    auto n_warps = n_nodes * n_split_targets * n_features;
    auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
    dh::LaunchKernel{n_blocks, kBlockThreads, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
        ScanHistogramKernel<kBlockThreads, decltype(d_scans)>, d_inputs, shared_inputs,
        d_sorted_idx, d_scans);
  }

  // Cache short-lived scratch to avoid synchronizing cudaFree calls between evaluation stages.
  dh::CachingDeviceUVector<MultiSplitCandidate> d_splits(n_nodes * n_features);
  {
    // One warp per block distributes features across more SMs for small node batches.
    std::uint32_t constexpr kBlockThreads = dh::WarpThreads();
    constexpr std::int32_t kWarpsPerBlk = kBlockThreads / dh::WarpThreads();
    auto n_warps = n_nodes * n_features;
    auto n_blocks = common::DivRoundUp(n_warps, kWarpsPerBlk);
    dh::LaunchKernel{n_blocks, kBlockThreads, 0, ctx->CUDACtx()->Stream()}(  // NOLINT
        EvaluateSplitsKernel<kBlockThreads, decltype(d_scans)>, d_inputs, shared_inputs, d_scans,
        evaluator, dh::ToSpan(d_splits));
  }

  // Find best split for each node
  auto d_weights = this->GetNodeWeights(n_split_targets);
  auto d_split_sums = this->split_sums_.View();
  auto d_split_cats = dh::ToSpan(this->split_cats_);
  auto node_cat_storage_size = this->node_cat_storage_size_;
  auto s_d_splits = dh::ToSpan(d_splits);

  // Reduce each node's feature candidates, preferring the smaller feature index on ties.
  auto offsets =
      dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) { return i * n_features; });
  dh::CachingDeviceUVector<MultiSplitCandidate> best_splits(out_splits.size());
  auto best = [] XGBOOST_DEVICE(MultiSplitCandidate const &lhs, MultiSplitCandidate const &rhs) {
    if (lhs.loss_chg > rhs.loss_chg) {
      return lhs;
    }
    if (rhs.loss_chg > lhs.loss_chg) {
      return rhs;
    }
    return lhs.findex <= rhs.findex ? lhs : rhs;
  };
  std::size_t temp_bytes = 0;
  dh::safe_cuda(cub::DeviceSegmentedReduce::Reduce(
      nullptr, temp_bytes, s_d_splits.data(), best_splits.data(), n_nodes, offsets, offsets + 1,
      best, MultiSplitCandidate{}, ctx->CUDACtx()->Stream()));
  dh::TemporaryArray<std::uint8_t> temp(temp_bytes);
  dh::safe_cuda(cub::DeviceSegmentedReduce::Reduce(
      temp.data().get(), temp_bytes, s_d_splits.data(), best_splits.data(), n_nodes, offsets,
      offsets + 1, best, MultiSplitCandidate{}, ctx->CUDACtx()->Stream()));
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
    QuantizedGradientSum parent_sum{input.parent_sum.data(), roundings};
    double parent_gain = evaluator.CalcGain(nidx, shared_inputs.param, parent_sum);
    double parent_hess = 0;
    for (bst_target_t t = 0; t < n_split_targets; ++t) {
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
    auto split_sum_dest = GetNodeSumImpl(d_split_sums, nidx, n_split_targets);
    QuantizedGradientSum child{split_sum.data(), roundings};
    QuantizedGradientDifference sibling{input.parent_sum.data(), split_sum.data(), roundings};

    double left_hess = 0, right_hess = 0;  // Sum of child hessians across all targets

    for (bst_target_t t = 0; t < n_split_targets; ++t) {
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

void GPUHistEvaluator::ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                                      common::Span<MultiExpandEntry const> h_candidates,
                                      common::Span<MultiExpandEntry const> d_candidates,
                                      bst_target_t n_split_targets) {
  xgboost_NVTX_FN_RANGE();
  CHECK_EQ(h_candidates.size(), d_candidates.size());
  CHECK(!h_candidates.empty());
  CHECK_EQ(n_split_targets, this->GetEvaluator().n_targets);

  auto weights = this->GetNodeWeights(n_split_targets);
  for (auto const &candidate : h_candidates) {
    auto left = p_tree->LeftChild(candidate.nidx);
    auto right = p_tree->RightChild(candidate.nidx);
    common::Span<float const> left_weight = weights.Left(candidate.nidx);
    common::Span<float const> right_weight = weights.Right(candidate.nidx);
    this->tree_evaluator_->AddSplit(candidate.nidx, left, right, candidate.split.findex,
                                    linalg::MakeVec(ctx->Device(), left_weight),
                                    linalg::MakeVec(ctx->Device(), right_weight));
  }

  // Assign the node sums here, for the next evaluate split call.
  auto mt_tree = MultiTargetTreeView{ctx->Device(), false, p_tree};
  auto max_in_it = dh::MakeIndexTransformIter([=] __device__(std::size_t i) -> bst_node_t {
    return std::max(mt_tree.LeftChild(d_candidates[i].nidx),
                    mt_tree.RightChild(d_candidates[i].nidx));
  });
  auto max_node = thrust::reduce(
      ctx->CUDACtx()->CTP(), max_in_it, max_in_it + d_candidates.size(), 0,
      [=] XGBOOST_DEVICE(bst_node_t l, bst_node_t r) { return cuda::std::max(l, r); });
  this->AllocNodeSum(max_node, n_split_targets);

  auto node_sums = this->node_sums_.View();
  // Use the internal split sums buffer instead of candidate.split.child_sum . It may be
  // stale in loss-guide grow policy (entries can remain in priority queue across
  // evaluation rounds).
  auto split_sums = this->split_sums_.View();

  dh::LaunchN(n_split_targets * d_candidates.size(), ctx->CUDACtx()->Stream(),
              [=] XGBOOST_DEVICE(std::size_t i) {
                auto get_node_sum = [&](bst_node_t nidx) {
                  return GetNodeSumImpl(node_sums, nidx, n_split_targets);
                };
                auto nidx_in_set = i / n_split_targets;
                auto t = i % n_split_targets;

                auto const &candidate = d_candidates[nidx_in_set];
                auto const &best_split = candidate.split;

                auto parent_sum = get_node_sum(candidate.nidx);
                // Look up split sum from persistent buffer by node id.
                // Use split_targets for indexing since that's what was used during storage.
                auto split_sum = GetNodeSumImpl(split_sums, candidate.nidx, n_split_targets);
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
void GPUHistEvaluator::Reset(Context const *ctx, common::HistogramCuts const &cuts,
                             common::Span<FeatureType const> feature_types,
                             bst_feature_t n_features, TrainParam const &param) {
  CHECK_EQ(n_features, cuts.cut_ptrs_.Size() - 1);
  this->Reset(ctx, cuts.cut_ptrs_.ConstDeviceSpan(), feature_types, param, 1);
  if (cuts.HasCategorical()) {
    this->node_cat_storage_size_ = common::CatBitField::ComputeStorageSize(cuts.MaxCategory() + 1);
  }
}

void GPUHistEvaluator::CopyToHost(Context const *ctx, std::vector<bst_node_t> const &nidx) {
  if (this->node_cat_storage_size_ == 0 || nidx.empty()) {
    return;
  }
  auto max_nidx = *std::max_element(nidx.cbegin(), nidx.cend());
  auto required = (max_nidx + 1) * node_cat_storage_size_;
  if (h_split_cats_.size() < required) {
    h_split_cats_.resize(required);
  }
  curt::Event event;
  event.Record(ctx->CUDACtx()->Stream());
  copy_stream_.View().Wait(event);
  for (auto node : nidx) {
    auto cats = this->GetNodeCats(node);
    dh::safe_cuda(cudaMemcpyAsync(h_split_cats_.data() + node * node_cat_storage_size_, cats.data(),
                                  cats.size_bytes(), cudaMemcpyDeviceToHost, copy_stream_.View()));
  }
}

void GPUHistEvaluator::EvaluateSplits(Context const *ctx, std::vector<bst_node_t> const &nidx,
                                      bst_feature_t max_active_features,
                                      common::Span<EvaluateSplitInputs const> inputs,
                                      EvaluateSplitSharedInputs shared,
                                      common::Span<GPUExpandEntry> out) {
  CHECK_EQ(inputs.size(), out.size());
  CHECK_EQ(inputs.size(), nidx.size());
  if (inputs.empty()) {
    return;
  }
  copy_stream_.View().Sync();
  if (max_active_features == 0) {
    auto evaluator = this->GetEvaluator();
    dh::LaunchN(inputs.size(), ctx->CUDACtx()->Stream(), [=] __device__(std::size_t i) {
      auto const &input = inputs[i];
      DeviceSplitCandidate split;
      split.left_sum = input.parent_sum;
      auto base = evaluator.CalcWeight(input.nidx, shared.param,
                                       shared.rounding.ToFloatingPoint(input.parent_sum));
      out[i] = {input.nidx, input.depth, split, base, 0.0f, 0.0f};
    });
    return;
  }

  dh::TemporaryArray<MultiEvaluateSplitInputs> node_storage(inputs.size());
  auto nodes = dh::ToSpan(node_storage);
  dh::TemporaryArray<GradientQuantiser> rounding_storage(1);
  auto roundings = dh::ToSpan(rounding_storage);
  dh::LaunchN(inputs.size(), ctx->CUDACtx()->Stream(), [=] __device__(std::size_t i) {
    // Pass the scalar quantizer through kernel arguments with the node inputs.
    if (i == 0) {
      roundings[0] = shared.rounding;
    }
    auto const &input = inputs[i];
    nodes[i] = {input.nidx,
                input.depth,
                {&input.parent_sum, 1},
                input.feature_set,
                input.gradient_histogram.subspan(0, shared.feature_values.size())};
  });
  MultiEvaluateSplitSharedInputs vector_shared;
  vector_shared.roundings = roundings;
  vector_shared.feature_segments = shared.feature_segments;
  vector_shared.feature_values = shared.feature_values.data();
  vector_shared.feature_types = shared.feature_types;
  vector_shared.cat_storage_size = this->node_cat_storage_size_;
  vector_shared.n_total_bins_per_tar = shared.feature_values.size();
  vector_shared.max_active_feature = max_active_features;
  vector_shared.param = shared.param;
  vector_shared.is_dense = shared.is_dense;

  dh::TemporaryArray<MultiExpandEntry> entry_storage(inputs.size());
  auto entries = dh::ToSpan(entry_storage);
  this->EvaluateSplits(ctx, nodes, vector_shared, *std::max_element(nidx.cbegin(), nidx.cend()),
                       entries);

  auto weights = this->GetNodeWeights(1);
  auto sums = this->split_sums_.View();
  dh::LaunchN(inputs.size(), ctx->CUDACtx()->Stream(), [=] __device__(std::size_t i) {
    auto const &entry = entries[i];
    auto const &candidate = entry.split;
    DeviceSplitCandidate split;
    split.loss_chg = candidate.loss_chg;
    split.dir = candidate.dir;
    split.findex = candidate.findex;
    split.is_cat = candidate.is_cat;
    split.fvalue = candidate.fvalue;
    split.thresh = candidate.thresh;
    // Scalar expansion entries use thresh for OHE as well as partition splits.
    if (split.is_cat && !isnan(split.fvalue)) {
      split.thresh = common::AsCat(split.fvalue);
      split.fvalue = std::numeric_limits<float>::quiet_NaN();
    }
    float left_weight = 0.0f, right_weight = 0.0f;
    if (candidate.child_sum.empty()) {
      split.left_sum = inputs[i].parent_sum;
    } else {
      auto child = sums[entry.nidx];
      auto sibling = inputs[i].parent_sum - child;
      split.left_sum = split.dir == kRightDir ? child : sibling;
      split.right_sum = split.dir == kRightDir ? sibling : child;
      left_weight = weights.Left(entry.nidx)[0];
      right_weight = weights.Right(entry.nidx)[0];
    }
    out[i] = {entry.nidx,  entry.depth, split, weights.Base(entry.nidx)[0],
              left_weight, right_weight};
  });
  this->CopyToHost(ctx, nidx);
}

GPUExpandEntry GPUHistEvaluator::EvaluateSingleSplit(Context const *ctx, EvaluateSplitInputs input,
                                                     EvaluateSplitSharedInputs shared) {
  dh::device_vector<EvaluateSplitInputs> inputs{input};
  dh::device_vector<GPUExpandEntry> entries(1);
  this->EvaluateSplits(ctx, {input.nidx}, input.feature_set.size(), dh::ToSpan(inputs), shared,
                       dh::ToSpan(entries));
  return entries[0];
}

void GPUHistEvaluator::ApplyTreeSplit(Context const *ctx, GPUExpandEntry const &candidate,
                                      RegTree *tree) {
  auto weights = this->GetNodeWeights(1);
  this->tree_evaluator_->AddSplit(candidate.nidx, tree->LeftChild(candidate.nidx),
                                  tree->RightChild(candidate.nidx), candidate.split.findex,
                                  linalg::MakeVec(ctx->Device(), weights.Left(candidate.nidx)),
                                  linalg::MakeVec(ctx->Device(), weights.Right(candidate.nidx)));
}
}  // namespace xgboost::tree::cuda_impl
