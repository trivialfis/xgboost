/**
 * Copyright 2017-2025, XGBoost contributors
 */
#include <thrust/extrema.h>    // for max_element
#include <thrust/sequence.h>   // for sequence
#include <thrust/transform.h>  // for transform

#include <cuda/functional>  // for proclaim_return_type
#include <limits>           // for numeric_limits
#include <vector>           // for vector

#include "../../common/cuda_context.cuh"    // for CUDAContext
#include "../../common/device_helpers.cuh"  // for CopyDeviceSpanToVector, ToSpan
#include "row_partitioner.cuh"

namespace xgboost::tree {
void RowPartitioner::Reset(Context const* ctx, bst_idx_t n_samples, bst_idx_t base_rowid) {
  ridx_segments_.clear();
  ridx_.resize(n_samples);
  tmp_.clear();
  n_nodes_ = 1;  // Root

  CHECK_LE(n_samples, std::numeric_limits<cuda_impl::RowIndexT>::max());
  ridx_segments_.emplace_back(
      NodePositionInfo{Segment{0, static_cast<cuda_impl::RowIndexT>(n_samples)}});

  thrust::sequence(ctx->CUDACtx()->CTP(), ridx_.data(), ridx_.data() + ridx_.size(), base_rowid);

  // Pre-allocate some host memory
  this->pinned_.GetSpan<std::int32_t>(1 << 11);
  this->pinned2_.GetSpan<std::int32_t>(1 << 13);
}

void RowPartitioner::Reset(Context const* ctx, common::Span<bst_idx_t const> ridx) {
  ridx_segments_.clear();
  ridx_.resize(ridx.size());
  tmp_.clear();
  n_nodes_ = 1;  // Root

  CHECK_LE(ridx.size(), std::numeric_limits<cuda_impl::RowIndexT>::max());
  ridx_segments_.emplace_back(
      NodePositionInfo{Segment{0, static_cast<cuda_impl::RowIndexT>(ridx.size())}});

  auto cuctx = ctx->CUDACtx();
  // Cast the global row indices (bst_idx_t) down to the 32-bit partitioner index type.
  thrust::transform(cuctx->CTP(), ridx.data(), ridx.data() + ridx.size(), ridx_.data(),
                    cuda::proclaim_return_type<cuda_impl::RowIndexT>(
                        [] __device__(bst_idx_t v) { return static_cast<cuda_impl::RowIndexT>(v); }));
  // Guard against row indices that exceed the 32-bit partitioner index space.
  if (!ridx.empty()) {
    auto max_it = thrust::max_element(cuctx->CTP(), ridx.data(), ridx.data() + ridx.size());
    bst_idx_t max_v = 0;
    dh::safe_cuda(cudaMemcpyAsync(&max_v, max_it, sizeof(bst_idx_t), cudaMemcpyDefault,
                                  cuctx->Stream()));
    cuctx->Stream().Sync();
    CHECK_LT(max_v, (static_cast<bst_idx_t>(1) << 32))
        << "Row index exceeds the 32-bit range supported by the row partitioner.";
  }

  // Pre-allocate some host memory
  this->pinned_.GetSpan<std::int32_t>(1 << 11);
  this->pinned2_.GetSpan<std::int32_t>(1 << 13);
}

RowPartitioner::~RowPartitioner() = default;

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows(bst_node_t nidx) {
  auto segment = ridx_segments_.at(nidx).segment;
  return dh::ToSpan(ridx_).subspan(segment.begin, segment.Size());
}

common::Span<const RowPartitioner::RowIndexT> RowPartitioner::GetRows() const {
  return dh::ToSpan(ridx_);
}

std::vector<RowPartitioner::RowIndexT> RowPartitioner::GetRowsHost(bst_node_t nidx) {
  auto span = GetRows(nidx);
  std::vector<RowIndexT> rows(span.size());
  dh::CopyDeviceSpanToVector(&rows, span);
  return rows;
}
};  // namespace xgboost::tree
