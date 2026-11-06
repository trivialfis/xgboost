/**
 * Copyright 2026, XGBoost Contributors
 * @brief Histogram scan and split evaluation shared by scalar and vector GPU trees.
 */
#pragma once

#include <cstdint>                   // for uint32_t, uint64_t
#include <cub/util_type.cuh>         // for KeyValuePair, Uninitialized
#include <cub/warp/warp_reduce.cuh>  // for WarpReduce
#include <cub/warp/warp_scan.cuh>    // for WarpScan
#include <cuda/ptx>                  // for get_sreg_laneid
#include <cuda/std/functional>       // for plus
#include <limits>                    // for numeric_limits
#include <type_traits>               // for integral_constant, is_same_v, is_trivially_copyable_v

#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "evaluate_splits.cuh"            // for EvaluateSplitInputs, MultiEvaluateSplitInputs

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

// The scan is fused with scoring. Only one running prefix per target is retained;
// the winning child sums are reconstructed once after all bins have been scored.
using ScanT = cub::WarpScan<GradientPairInt64>;
using ArgMaxT = cub::KeyValuePair<std::uint32_t, std::uint64_t>;
using MaxT = cub::WarpReduce<ArgMaxT>;
using SumT = cub::WarpReduce<GradientPairInt64>;

// Compare double gains without floating-point instructions, preserving cub::ArgMax's
// lower-index tie breaking and unordered-comparison behavior.
struct IntegerArgMax {
  // Encode once per lane, before reduction. The original gain stays in that lane.
  static __device__ std::uint64_t Encode(double gain) {
    auto bits = static_cast<std::uint64_t>(__double_as_longlong(gain));
    constexpr std::uint64_t kSign = std::uint64_t{1} << 63;
    constexpr std::uint64_t kMagnitude = kSign - 1;
    constexpr std::uint64_t kInfinity = 0x7ff0000000000000ULL;
    // No non-NaN value maps to zero; reserve it for NaNs.
    if ((bits & kMagnitude) > kInfinity) {
      return 0;
    }
    // Signed zeros compare equal.
    if ((bits & kMagnitude) == 0) {
      bits = 0;
    }
    // Reverse negative encodings and put nonnegative encodings above them.
    return bits ^ ((bits & kSign) ? ~std::uint64_t{0} : kSign);
  }

  __device__ ArgMaxT operator()(ArgMaxT const &a, ArgMaxT const &b) const {
    // Both comparisons in cub::ArgMax are false if either operand is NaN.
    if (a.value == 0 || b.value == 0) {
      return a;
    }
    return b.value > a.value || (b.value == a.value && b.key < a.key) ? b : a;
  }
};

enum class SplitKind { kNumerical, kOneHot, kPartition };

template <SplitKind kKind, DefaultDirection kDirection, typename SortedIndex>
__device__ bst_bin_t HistogramIndex(bst_bin_t begin, bst_bin_t end, bst_bin_t bin,
                                    SortedIndex sorted_idx, std::size_t sorted_offset) {
  if constexpr (kKind == SplitKind::kPartition) {
    auto index = kDirection == kLeftDir ? bin : RevBinIdx(begin, end, bin);
    return sorted_idx[sorted_offset + index];
  } else if constexpr (kKind == SplitKind::kNumerical) {
    return kDirection == kRightDir ? bin : RevBinIdx(begin, end, bin);
  } else {
    return bin;
  }
}

__device__ inline GradientPairInt64 Broadcast(GradientPairInt64 value) {
  return {__shfl_sync(dh::WarpFullMask(), static_cast<long long>(value.GetQuantisedGrad()), 0),
          __shfl_sync(dh::WarpFullMask(), static_cast<long long>(value.GetQuantisedHess()), 0)};
}

template <SplitKind kKind, DefaultDirection kDirection, typename SortedIndex>
__device__ void Enumerate(MultiEvaluateSplitInputs const &node,
                          MultiEvaluateSplitSharedInputs const &shared, bst_feature_t fidx,
                          TreeEvaluator::SplitEvaluator<EvalParam> const &evaluator,
                          SortedIndex sorted_idx, std::size_t sorted_offset,
                          common::Span<GradientPairInt64> carry, ScanT::TempStorage *scan_storage,
                          MaxT::TempStorage *max_storage, SumT::TempStorage *sum_storage,
                          MultiSplitCandidate *best, bst_bin_t *best_bin) {
  auto lane = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
  bst_bin_t begin = shared.feature_segments[fidx], end = shared.feature_segments[fidx + 1];
  auto nt = shared.Targets();
  auto limit = end;
  if constexpr (kKind == SplitKind::kPartition) {
    limit = begin + std::min(shared.param.max_cat_threshold, end - begin) - 1;
  }
  for (bst_bin_t tile = begin; tile < limit; tile += dh::WarpThreads()) {
    auto bin_idx = tile + lane;
    auto hidx = HistogramIndex<kKind, kDirection>(begin, end, std::min(bin_idx, end - 1),
                                                  sorted_idx, sorted_offset);
    double gain = 0, left_hess = 0, right_hess = 0;
    for (bst_target_t t = 0; t < nt; ++t) {
      auto hist = node.histogram.data() + t * shared.n_total_bins_per_tar;
      auto bin = bin_idx < limit ? hist[hidx] : GradientPairInt64{};
      if constexpr (kKind == SplitKind::kOneHot) {
        if constexpr (kDirection == kRightDir) {
          // Missing values follow the matching category right. Cache the observed
          // feature sum once per target to obtain the left (other categories) sum.
          GradientPairInt64 total;
          if (tile == begin) {
            for (bst_bin_t i = begin + lane; i < end; i += dh::WarpThreads()) {
              total += hist[i];
            }
            total = SumT(*sum_storage).Sum(total);
            __syncwarp();
            total = Broadcast(total);
            if (lane == 0) {
              carry[t] = total;
            }
          } else {
            total = carry[t];
          }
          bin = total - bin;
        }
      } else {
        if (lane == 0 && tile != begin) {
          bin += carry[t];
        }
        GradientPairInt64 aggregate;
        ScanT(*scan_storage).InclusiveScan(bin, bin, cuda::std::plus{}, aggregate);
        __syncwarp();
        if (lane == 0) {
          carry[t] = aggregate;
        }
      }
      auto child = shared.roundings[t].ToFloatingPoint(bin);
      auto sibling = shared.roundings[t].ToFloatingPoint(node.parent_sum[t] - bin);
      auto left = kDirection == kRightDir ? child : sibling;
      auto right = kDirection == kRightDir ? sibling : child;
      left_hess += left.GetHess();
      right_hess += right.GetHess();
      // Preserve the target order and scoring used by the shared constraint evaluator.
      if (evaluator.has_constraint) {
        evaluator.AccumulateConstrainedSplitGain(shared.param, node.nidx, fidx, t, left, right,
                                                 nt == 1, &gain);
      } else {
        gain += tree::CalcGain(shared.param, left.GetGrad(), left.GetHess());
        gain += tree::CalcGain(shared.param, right.GetGrad(), right.GetHess());
      }
    }
    if (bin_idx >= limit || !IsValidSplit(shared.param, left_hess / nt, right_hess / nt)) {
      gain = -std::numeric_limits<double>::infinity();
    }
    auto max = MaxT(*max_storage)
                   .Reduce({static_cast<std::uint32_t>(lane), IntegerArgMax::Encode(gain)},
                           IntegerArgMax{});
    auto winner = __shfl_sync(dh::WarpFullMask(), max.key, 0);
    if (lane == winner && !isinf(gain) && static_cast<float>(gain) > best->loss_chg) {
      *best_bin = bin_idx;
      if constexpr (kKind == SplitKind::kPartition) {
        auto offset = bin_idx - begin;
        auto threshold = kDirection == kLeftDir ? offset : end - begin - offset - 1;
        best->UpdateCat(gain, kDirection, threshold, fidx, carry);
      } else {
        auto value = shared.feature_values[bin_idx];
        if constexpr (kKind == SplitKind::kNumerical && kDirection == kLeftDir) {
          auto split_bin = RevBinIdx(begin, end, bin_idx);
          value = split_bin == begin ? -std::numeric_limits<float>::infinity()
                                     : shared.feature_values[split_bin - 1];
        }
        best->Update(gain, kDirection, value, fidx, carry, kKind == SplitKind::kOneHot);
      }
    }
    __syncwarp();
  }
}

template <SplitKind kKind, typename SortedIndex>
__device__ void RestoreWinner(MultiEvaluateSplitInputs const &node,
                              MultiEvaluateSplitSharedInputs const &shared, bst_feature_t fidx,
                              SortedIndex sorted_idx, std::size_t sorted_offset,
                              common::Span<GradientPairInt64> sums, SumT::TempStorage *storage,
                              MultiSplitCandidate const &best, bst_bin_t best_bin) {
  if (best_bin < 0) {
    return;
  }
  auto lane = static_cast<bst_bin_t>(cuda::ptx::get_sreg_laneid());
  bst_bin_t begin = shared.feature_segments[fidx], end = shared.feature_segments[fidx + 1];
  for (bst_target_t t = 0; t < shared.Targets(); ++t) {
    auto hist = node.histogram.data() + t * shared.n_total_bins_per_tar;
    GradientPairInt64 local;
    if constexpr (kKind == SplitKind::kOneHot) {
      if (best.dir == kLeftDir) {
        if (lane == 0) {
          local = hist[best_bin];
        }
      } else {
        for (bst_bin_t i = begin + lane; i < end; i += dh::WarpThreads()) {
          if (i != best_bin) {
            local += hist[i];
          }
        }
      }
    } else {
      for (bst_bin_t i = begin + lane; i <= best_bin; i += dh::WarpThreads()) {
        auto hidx = best.dir == kRightDir
                        ? HistogramIndex<kKind, kRightDir>(begin, end, i, sorted_idx, sorted_offset)
                        : HistogramIndex<kKind, kLeftDir>(begin, end, i, sorted_idx, sorted_offset);
        local += hist[hidx];
      }
    }
    auto sum = SumT(*storage).Sum(local);
    __syncwarp();
    if (lane == 0) {
      sums[t] = sum;
    }
  }
  __syncwarp();
}

// Common target counts expose loop bounds to the compiler. The default retains
// the runtime target count and uses the same fused kernel for every other size.
template <typename Fn>
void DispatchFusedTargets(bst_target_t n_targets, Fn &&fn) {
  switch (n_targets) {
    case 1:
      fn(std::integral_constant<bst_target_t, 1>{});
      break;
    case 2:
      fn(std::integral_constant<bst_target_t, 2>{});
      break;
    case 3:
      fn(std::integral_constant<bst_target_t, 3>{});
      break;
    case 4:
      fn(std::integral_constant<bst_target_t, 4>{});
      break;
    case 8:
      fn(std::integral_constant<bst_target_t, 8>{});
      break;
    case 16:
      fn(std::integral_constant<bst_target_t, 16>{});
      break;
    case 32:
      fn(std::integral_constant<bst_target_t, 32>{});
      break;
    case 64:
      fn(std::integral_constant<bst_target_t, 64>{});
      break;
    case 128:
      fn(std::integral_constant<bst_target_t, 128>{});
      break;
    default:
      fn(std::integral_constant<bst_target_t, 0>{});
      break;
  }
}

template <bst_target_t kTargets, bool kHasCategorical = true, typename Input, typename SharedInput,
          typename SortedIndex, typename Output>
__global__ __launch_bounds__(32) void EvaluateSplitsKernel(
    common::Span<Input const> nodes, SharedInput shared_input, bst_feature_t features,
    SortedIndex sorted_idx, TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
    common::Span<GradientPairInt64> scratch, common::Span<Output> out) {
  auto shared = GetSharedInput(shared_input, features);
  if constexpr (kTargets != 0) {
    shared.roundings = {shared.roundings.data(), kTargets};
  }
  auto nidx = blockIdx.x / features;
  auto node = GetInput(nodes[nidx]);
  auto fi = blockIdx.x % features;
  auto lane = cuda::ptx::get_sreg_laneid();
  if (fi >= node.feature_set.size()) {
    if (lane == 0) {
      out[blockIdx.x] = Output{};
    }
    return;
  }
  auto fidx = node.feature_set[fi];
  __shared__ ScanT::TempStorage scan_storage;
  __shared__ MaxT::TempStorage max_storage;
  __shared__ SumT::TempStorage sum_storage;
  __shared__ cub::Uninitialized<MultiSplitCandidate> split_storage;
  __shared__ cub::Uninitialized<GradientPairInt64> scalar_scratch;
  __shared__ bst_bin_t best_bin;
  auto &best = split_storage.Alias();
  auto nt = shared.Targets();
  common::Span<GradientPairInt64> carry;
  if constexpr (std::is_same_v<Output, DeviceSplitCandidate>) {
    carry = {&scalar_scratch.Alias(), 1};
  } else {
    carry = scratch.subspan(blockIdx.x * nt, nt);
  }
  if (lane == 0) {
    best = MultiSplitCandidate{};
    best_bin = -1;
  }
  __syncwarp();
  auto sorted_offset = static_cast<std::size_t>(nidx) * shared.n_total_bins_per_tar;
  auto enumerate = [&](auto kind, auto direction) {
    Enumerate<decltype(kind)::value, decltype(direction)::value>(
        node, shared, fidx, evaluator, sorted_idx, sorted_offset, carry, &scan_storage,
        &max_storage, &sum_storage, &best, &best_bin);
  };
  auto restore = [&](auto kind) {
    RestoreWinner<decltype(kind)::value>(node, shared, fidx, sorted_idx, sorted_offset, carry,
                                         &sum_storage, best, best_bin);
  };
  auto left = std::integral_constant<DefaultDirection, kLeftDir>{};
  auto right = std::integral_constant<DefaultDirection, kRightDir>{};
  bool is_categorical = false;
  if constexpr (kHasCategorical) {
    is_categorical = shared.IsCategorical(fidx);
  }
  if (is_categorical) {
    if constexpr (kHasCategorical) {
      auto bins = shared.feature_segments[fidx + 1] - shared.feature_segments[fidx];
      if (common::UseOneHot(bins, shared.param.max_cat_to_onehot)) {
        auto kind = std::integral_constant<SplitKind, SplitKind::kOneHot>{};
        enumerate(kind, right);
        enumerate(kind, left);
        restore(kind);
      } else {
        auto kind = std::integral_constant<SplitKind, SplitKind::kPartition>{};
        enumerate(kind, left);
        enumerate(kind, right);
        restore(kind);
      }
    }
  } else {
    auto kind = std::integral_constant<SplitKind, SplitKind::kNumerical>{};
    if (shared.one_pass != MultiEvaluateSplitSharedInputs::kBackward) {
      enumerate(kind, right);
    }
    if (shared.NeedBackward()) {
      enumerate(kind, left);
    }
    restore(kind);
  }
  if (lane == 0) {
    StoreCandidate(best, node, &out[blockIdx.x]);
  }
}
// Both host evaluators use this launch path; input/output adapters stay in the kernel.
template <typename Input, typename SharedInput, typename SortedIndex, typename Output>
void LaunchEvaluateSplits(Context const *ctx, common::Span<Input const> nodes, SharedInput shared,
                          bst_feature_t features, SortedIndex sorted,
                          TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
                          common::Span<GradientPairInt64> scratch, common::Span<Output> out,
                          bool has_categorical) {
  auto launch = [&](auto targets, auto categorical) {
    dh::LaunchKernel{static_cast<std::uint32_t>(nodes.size() * features), 32, 0,
                     ctx->CUDACtx()->Stream()}(
        EvaluateSplitsKernel<decltype(targets)::value, decltype(categorical)::value, Input,
                             SharedInput, SortedIndex, Output>,
        nodes, shared, features, sorted, evaluator, scratch, out);
  };
  auto dispatch = [&](auto targets) {
    if (has_categorical) {
      launch(targets, std::true_type{});
    } else {
      launch(targets, std::false_type{});
    }
  };
  if constexpr (std::is_same_v<Input, EvaluateSplitInputs>) {
    dispatch(std::integral_constant<bst_target_t, 1>{});
  } else {
    DispatchFusedTargets(shared.Targets(), dispatch);
  }
}
}  // namespace split_kernel
}  // namespace xgboost::tree::cuda_impl
