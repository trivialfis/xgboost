/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <algorithm>                               // for copy, max, transform
#include <cub/device/device_segmented_reduce.cuh>  // for DeviceSegmentedReduce
#include <vector>                                  // for vector

#include "../../common/categorical.h"
#include "../../common/cuda_compat.cuh"   // for CUDA compatibility
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/cuda_stream.h"     // for Event
#include "../../common/nvtx_utils.h"      // for xgboost_NVTX_FN_RANGE
#include "evaluate_splits.cuh"
#include "evaluate_splits_impl.cuh"
#include "expand_entry.cuh"
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::tree {
void GPUHistEvaluator::PrepareTreeSplit(Context const *ctx,
                                        common::Span<GPUExpandEntry const> candidates) {
  if (!tree_evaluator_.HasConstraint() || candidates.size() <= 1) {
    return;
  }
  auto h_candidates = h_split_candidates_.GetSpan<GPUExpandEntry>(candidates.size());
  std::copy(candidates.begin(), candidates.end(), h_candidates.begin());
  // Queue the upload before host tree expansion. The bounds update uses the same stream.
  dh::CopyTo(h_candidates, &d_split_candidates_, ctx->CUDACtx()->Stream());
}

void GPUHistEvaluator::ApplyTreeSplit(Context const *ctx, RegTree const &tree,
                                      common::Span<GPUExpandEntry const> candidates) {
  xgboost_NVTX_FN_RANGE();
  CHECK(!candidates.empty());
  if (!tree_evaluator_.HasConstraint()) {
    return;
  }
  if (candidates.size() == 1) {
    // Use kernel arguments for a singleton batch, avoiding both explicit uploads.
    auto const &candidate = candidates.front();
    TreeEvaluator::SplitBounds split{candidate.nidx,
                                     tree.LeftChild(candidate.nidx),
                                     tree.RightChild(candidate.nidx),
                                     static_cast<bst_feature_t>(candidate.split.findex),
                                     candidate.left_weight,
                                     candidate.right_weight};
    tree_evaluator_.AddSplits(1, std::max(split.left_nidx, split.right_nidx),
                              [=] XGBOOST_DEVICE(std::size_t, bst_target_t) { return split; });
    return;
  }

  CHECK_EQ(d_split_candidates_.size(), candidates.size());
  auto h_children = h_split_children_.GetSpan<ChildNodes>(candidates.size());
  bst_node_t max_nidx{0};
  std::transform(candidates.begin(), candidates.end(), h_children.begin(),
                 [&](auto const &candidate) {
                   auto left = tree.LeftChild(candidate.nidx);
                   auto right = tree.RightChild(candidate.nidx);
                   max_nidx = std::max({max_nidx, left, right});
                   return ChildNodes{left, right};
                 });
  // Upload only the selected children instead of materializing the whole device tree.
  dh::CopyTo(h_children, &d_split_children_, ctx->CUDACtx()->Stream());
  common::Span<GPUExpandEntry const> d_candidates = dh::ToSpan(d_split_candidates_);
  common::Span<ChildNodes const> d_children = dh::ToSpan(d_split_children_);
  tree_evaluator_.AddSplits(
      d_candidates.size(), max_nidx, [=] XGBOOST_DEVICE(std::size_t i, bst_target_t) {
        auto const &candidate = d_candidates[i];
        return TreeEvaluator::SplitBounds{
            candidate.nidx,        d_children[i].first,
            d_children[i].second,  static_cast<bst_feature_t>(candidate.split.findex),
            candidate.left_weight, candidate.right_weight};
      });
}

__device__ DeviceSplitCandidate operator+(const DeviceSplitCandidate &a,
                                          const DeviceSplitCandidate &b) {
  return b.loss_chg > a.loss_chg ? b : a;
}

/**
 * \brief Set the bits for categorical splits based on the split threshold.
 */
__device__ void SetCategoricalSplit(const EvaluateSplitSharedInputs &shared_inputs,
                                    common::Span<bst_feature_t const> d_sorted_idx,
                                    bst_feature_t fidx, std::size_t input_idx,
                                    common::Span<common::CatBitField::value_type> out,
                                    DeviceSplitCandidate *p_out_split) {
  auto &out_split = *p_out_split;
  auto out_cats = common::CatBitField{out};

  // Simple case for one hot split
  if (common::UseOneHot(shared_inputs.FeatureBins(fidx), shared_inputs.param.max_cat_to_onehot)) {
    out_cats.Set(common::AsCat(out_split.thresh));
    return;
  }

  // partition-based split
  auto node_sorted_idx = d_sorted_idx.subspan(shared_inputs.feature_values.size() * input_idx,
                                              shared_inputs.feature_values.size());
  size_t node_offset = input_idx * shared_inputs.feature_values.size();
  auto const best_thresh = out_split.thresh;
  if (best_thresh == -1) {
    return;
  }
  auto f_sorted_idx = node_sorted_idx.subspan(shared_inputs.feature_segments[fidx],
                                              shared_inputs.FeatureBins(fidx));
  bool forward = out_split.dir == kLeftDir;
  bst_bin_t partition = forward ? best_thresh + 1 : best_thresh;
  auto beg = dh::tcbegin(f_sorted_idx);
  assert(partition > 0 && "Invalid partition.");
  thrust::for_each(thrust::seq, beg, beg + partition, [&](size_t c) {
    auto cat = shared_inputs.feature_values[c - node_offset];
    out_cats.Set(common::AsCat(cat));
  });
}

void GPUHistEvaluator::LaunchEvaluateSplits(Context const *ctx, bst_feature_t max_active_features,
                                            common::Span<const EvaluateSplitInputs> d_inputs,
                                            EvaluateSplitSharedInputs shared_inputs,
                                            TreeEvaluator::SplitEvaluator<EvalParam> evaluator,
                                            common::Span<DeviceSplitCandidate> out_splits) {
  if (need_sort_histogram_) {
    this->SortHistogram(ctx, d_inputs, shared_inputs, evaluator);
  }

  size_t combined_num_features = max_active_features * d_inputs.size();
  // Every block writes its candidate, no initialization is needed.
  dh::TemporaryArray<DeviceSplitCandidate> feature_best_splits(combined_num_features);

  if (!has_categoricals_) {
    dh::LaunchKernel{static_cast<std::uint32_t>(combined_num_features), 32, 0,
                     ctx->CUDACtx()->Stream()}(
        cuda_impl::split_kernel::FusedEvaluateSplitsKernel<1, EvaluateSplitInputs,
                                                         EvaluateSplitSharedInputs,
                                                         DeviceSplitCandidate>,
        d_inputs, shared_inputs, max_active_features, evaluator,
        common::Span<GradientPairInt64>{}, dh::ToSpan(feature_best_splits));
  } else {
    auto node_hist_size = shared_inputs.feature_values.size();
    scan_buffer_.resize(d_inputs.size() * node_hist_size * 2);
    auto scans = dh::ToSpan(scan_buffer_);
    auto d_scans =
        common::IterSpan{dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
                           return scans.subspan(i * node_hist_size * 2, node_hist_size * 2);
                         }),
                         d_inputs.size()};
    // Scalar category sorting stores global offsets; the shared scan uses node-local bins.
    auto sorted = this->SortedIdx(d_inputs.size(), node_hist_size);
    auto sorted_idx = common::IterSpan{dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) {
                                         return sorted[i] - (i / node_hist_size) * node_hist_size;
                                       }),
                                       sorted.size()};

    constexpr std::uint32_t kScanThreads = 128;
    constexpr auto kWarpsPerBlock = kScanThreads / dh::WarpThreads();
    auto n_blocks = common::DivRoundUp(combined_num_features, kWarpsPerBlock);
    dh::LaunchKernel{static_cast<std::uint32_t>(n_blocks), kScanThreads, 0, ctx->CUDACtx()->Stream()}(
        cuda_impl::split_kernel::ScanHistogramKernel<kScanThreads, EvaluateSplitInputs,
                                                     EvaluateSplitSharedInputs, decltype(sorted_idx),
                                                     decltype(d_scans)>,
        d_inputs, shared_inputs, max_active_features, sorted_idx, d_scans);

    constexpr std::uint32_t kEvalThreads = 32;
    dh::LaunchKernel{static_cast<std::uint32_t>(combined_num_features), kEvalThreads, 0,
                     ctx->CUDACtx()->Stream()}(
        cuda_impl::split_kernel::EvaluateSplitsKernel<kEvalThreads, EvaluateSplitInputs,
                                                      EvaluateSplitSharedInputs, decltype(d_scans),
                                                      DeviceSplitCandidate>,
        d_inputs, shared_inputs, max_active_features, d_scans, evaluator,
        dh::ToSpan(feature_best_splits));

  }

  // Reduce to get best candidate for left and right child over all features
  auto reduce_offset = dh::MakeTransformIterator<size_t>(
      dh::make_counting_iterator(0llu),
      [=] __device__(size_t idx) -> size_t { return idx * max_active_features; });
  size_t temp_storage_bytes = 0;
  auto num_segments = out_splits.size();
  dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(
      nullptr, temp_storage_bytes, feature_best_splits.data(), out_splits.data(), num_segments,
      reduce_offset, reduce_offset + 1, ctx->CUDACtx()->Stream()));
  dh::TemporaryArray<int8_t> temp(temp_storage_bytes);
  dh::safe_cuda(cub::DeviceSegmentedReduce::Sum(
      temp.data().get(), temp_storage_bytes, feature_best_splits.data(), out_splits.data(),
      num_segments, reduce_offset, reduce_offset + 1, ctx->CUDACtx()->Stream()));
}

void GPUHistEvaluator::CopyToHost(const std::vector<bst_node_t> &nidx) {
  if (!has_categoricals_) return;
  auto d_cats = this->DeviceCatStorage(nidx);
  auto h_cats = this->HostCatStorage(nidx);
  curt::Event event;
  event.Record(curt::DefaultStream());
  for (auto idx : nidx) {
    copy_stream_.View().Wait(event);
    dh::safe_cuda(cudaMemcpyAsync(
        h_cats.GetNodeCatStorage(idx).data(), d_cats.GetNodeCatStorage(idx).data(),
        d_cats.GetNodeCatStorage(idx).size_bytes(), cudaMemcpyDeviceToHost, copy_stream_.View()));
  }
}

void GPUHistEvaluator::EvaluateSplits(Context const *ctx, const std::vector<bst_node_t> &nidx,
                                      bst_feature_t max_active_features,
                                      common::Span<const EvaluateSplitInputs> d_inputs,
                                      EvaluateSplitSharedInputs shared_inputs,
                                      common::Span<GPUExpandEntry> out_entries) {
  auto evaluator = this->tree_evaluator_.template GetEvaluator<EvalParam>();

  dh::TemporaryArray<DeviceSplitCandidate> splits_out_storage(d_inputs.size());
  auto out_splits = dh::ToSpan(splits_out_storage);
  this->LaunchEvaluateSplits(ctx, max_active_features, d_inputs, shared_inputs, evaluator,
                             out_splits);

  auto d_sorted_idx = this->SortedIdx(d_inputs.size(), shared_inputs.feature_values.size());
  auto d_entries = out_entries;
  auto device_cats_accessor = this->DeviceCatStorage(nidx);
  // turn candidate into entry, along with handling sort based split.
  dh::LaunchN(d_inputs.size(), ctx->CUDACtx()->Stream(), [=] __device__(size_t i) mutable {
    auto const input = d_inputs[i];
    auto &split = out_splits[i];
    // Subtract parent gain here
    // As it is constant, this is more efficient than doing it during every
    // split evaluation
    float parent_gain = evaluator.CalcGain(
        input.nidx, shared_inputs.param, shared_inputs.rounding.ToFloatingPoint(input.parent_sum));
    split.loss_chg -= parent_gain;
    auto fidx = out_splits[i].findex;

    if (split.is_cat) {
      SetCategoricalSplit(shared_inputs, d_sorted_idx, fidx, i,
                          device_cats_accessor.GetNodeCatStorage(input.nidx), &out_splits[i]);
    }

    float base_weight = evaluator.CalcWeight(
        input.nidx, shared_inputs.param,
        shared_inputs.rounding.ToFloatingPoint(split.left_sum + split.right_sum));
    float left_weight = evaluator.CalcWeight(
        input.nidx, shared_inputs.param, shared_inputs.rounding.ToFloatingPoint(split.left_sum));
    float right_weight = evaluator.CalcWeight(
        input.nidx, shared_inputs.param, shared_inputs.rounding.ToFloatingPoint(split.right_sum));

    d_entries[i] = GPUExpandEntry{input.nidx,  input.depth, out_splits[i],
                                  base_weight, left_weight, right_weight};
  });

  this->CopyToHost(nidx);
}

GPUExpandEntry GPUHistEvaluator::EvaluateSingleSplit(Context const *ctx, EvaluateSplitInputs input,
                                                     EvaluateSplitSharedInputs shared_inputs) {
  dh::CachingDeviceUVector<EvaluateSplitInputs> inputs(1);
  dh::safe_cuda(cudaMemcpyAsync(inputs.data(), &input, sizeof(input), cudaMemcpyDefault));

  dh::TemporaryArray<GPUExpandEntry> out_entries(1);
  this->EvaluateSplits(ctx, {input.nidx}, input.feature_set.size(), dh::ToSpan(inputs),
                       shared_inputs, dh::ToSpan(out_entries));
  GPUExpandEntry root_entry;
  dh::safe_cuda(cudaMemcpyAsync(&root_entry, out_entries.data().get(), sizeof(GPUExpandEntry),
                                cudaMemcpyDeviceToHost));
  return root_entry;
}
}  // namespace xgboost::tree
