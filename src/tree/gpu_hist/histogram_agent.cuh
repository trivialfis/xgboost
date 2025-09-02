/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include <cuda/pipeline>

#include "feature_groups.cuh"
#include "quantiser.cuh"
#include "row_partitioner.cuh"
#include "xgboost/base.h"

namespace xgboost::tree::cuda_impl {
XGBOOST_DEV_INLINE void AtomicAddGpairGlobal(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<uint64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  atomicAdd(dst_ptr, *reinterpret_cast<uint64_t*>(&g));
  atomicAdd(dst_ptr + 1, *reinterpret_cast<uint64_t*>(&h));
}

template <typename Accessor, bool kCompressed, bool kDense, int kBlockThreads, int kItemsPerThread>
class HistogramAgent {
  XGBOOST_DEV_INLINE static bst_idx_t IterIdx(Accessor const& matrix,
                                              RowPartitioner::RowIndexT ridx, bst_feature_t fidx) {
    // ridx_local = ridx - base_rowid  <== Row index local to each batch
    // entry_idx = ridx_local * row_stride <== Starting entry index for this row in the matrix
    // entry_idx += start_feature  <== Inside a row, first column inside this feature group
    // idx % feature_stride <== The feaature index local to the current feature group
    // entry_idx += idx % feature_stride <== Final index.
    return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
  }
  XGBOOST_DEV_INLINE static bst_feature_t FeatIdx(FeatureGroup const& group, bst_idx_t idx,
                                                  std::int32_t feature_stride) {
    auto fidx = group.start_feature + idx % feature_stride;
    return fidx;
  }

  int constexpr static kItemsPerTile = kBlockThreads * kItemsPerThread;

  GradientPairInt64* smem_arr_;
  GradientPairInt64* d_node_hist_;
  using Idx = cuda_impl::RowIndexT;

  dh::LDGIterator<const Idx> d_ridx_;
  const GradientPair* d_gpair_;
  const FeatureGroup group_;
  Accessor const& matrix_;
  const int feature_stride_;
  const bst_idx_t n_elements_;
  const GradientQuantiser& rounding_;

  static_assert(kCompressed >= kDense);

 public:
  __device__ HistogramAgent(GradientPairInt64* smem_arr,
                            GradientPairInt64* __restrict__ d_node_hist, const FeatureGroup& group,
                            Accessor const& matrix, common::Span<const Idx> d_ridx,
                            const GradientQuantiser& rounding, const GradientPair* d_gpair)
      : smem_arr_{smem_arr},
        d_node_hist_{d_node_hist},
        d_ridx_(d_ridx.data()),
        group_{group},
        matrix_(matrix),
        feature_stride_(kCompressed ? group.num_features : matrix.row_stride),
        n_elements_{feature_stride_ * d_ridx.size()},
        rounding_{rounding},
        d_gpair_{d_gpair} {}

  template <typename Fn>
  __device__ void ProcessPartialTileShared(std::size_t offset, Fn&& fn) {
    for (std::size_t idx = offset + threadIdx.x,
                     n = std::min(offset + kBlockThreads * kItemsPerTile, n_elements_);
         idx < n; idx += kBlockThreads) {
      Idx ridx = d_ridx_[idx / feature_stride_];
      auto fidx = FeatIdx(group_, idx, feature_stride_);
      bst_bin_t compressed_bin = matrix_.gidx_iter[IterIdx(matrix_, ridx, fidx)];
      if (kDense || compressed_bin != matrix_.NullValue()) {
        // The matrix is compressed with feature-local bins.
        if (kCompressed) {
          compressed_bin += this->matrix_.feature_segments[fidx];
        }
        // Avoid atomic add if it's a null value.
        auto adjusted = rounding_.ToFixedPoint(d_gpair_[ridx]);
        // Subtract start_bin to write to group-local histogram. If this is not a dense
        // matrix, then start_bin is 0 since featuregrouping doesn't support sparse data.

        // AtomicAddGpairShared
        fn(compressed_bin - group_.start_bin, adjusted);
      }
    }
  }

  // Instruction level parallelism by loop unrolling
  // Allows the kernel to pipeline many operations while waiting for global memory
  // Increases the throughput of this kernel significantly
  template <typename Fn, typename Gfn>
  __device__ void BuildHistogramWithShared(Fn&& fn, Gfn&& gfn) {
    dh::BlockFill(smem_arr_, group_.num_bins, GradientPairInt64{});
    __syncthreads();

    std::size_t offset = blockIdx.x * kItemsPerTile;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    constexpr int kStages = 2;
    int constexpr kStageSize = kItemsPerThread / kStages;
    std::size_t idx_s[kStages][kStageSize];
    Idx ridx_s[kStages][kStageSize];
    GradientPair gpair_s[kStages][kStageSize];
    bst_bin_t gidx_s[kStages][kStageSize];

    auto load = [this](std::size_t(&idx)[kStageSize], Idx(&ridx)[kStageSize],
                       GradientPair(&gpair)[kStageSize], int stage, std::size_t offset) {
#pragma unroll
      for (int i = 0; i < kStageSize; i++) {
        auto k = stage * kStageSize + i;
        idx[i] = offset + k * kBlockThreads + threadIdx.x;
      }
#pragma unroll
      for (int i = 0; i < kStageSize; i++) {
        ridx[i] = d_ridx_[idx[i] / feature_stride_];
      }
#pragma unroll
      for (int i = 0; i < kStageSize; i++) {
        gpair[i] = d_gpair_[ridx[i]];
      }
    };
    auto stage_buf = reinterpret_cast<unsigned char*>(smem_arr_);  // fixme: type

    constexpr int kBufSize = 5;
    auto load_gidx = [&, this](std::size_t(&idx)[kStageSize], Idx(&ridx)[kStageSize],
                               bst_bin_t(&gidx)[kStageSize], int stage) {
#pragma unroll
      for (int i = 0; i < kStageSize; i++) {
        auto fidx = FeatIdx(group_, idx[i], feature_stride_);
        auto itidx = IterIdx(matrix_, ridx[i], fidx);

        auto shmem_beg_idx = kBlockThreads * i * kBufSize + (threadIdx.x * kBufSize);
        shmem_beg_idx = shmem_beg_idx + stage * kBlockThreads * kStageSize * kBufSize;
        matrix_.gidx_iter.LoadBuf(itidx, stage_buf + shmem_beg_idx, pipe);

        gidx[i] = matrix_.gidx_iter[IterIdx(matrix_, ridx[i], fidx)];
        if (kDense || gidx[i] != matrix_.NullValue()) {
          if constexpr (kCompressed) {
            gidx[i] += matrix_.feature_segments[fidx];
          }
        } else {
          // Use -1 to denote missing. Since we need to add the beginning bin to gidx, the
          // result might equal to the `NullValue`.
          gidx[i] = -1;
        }
      }
    };

    auto write_gidx = [this, fn, stage_buf](std::size_t(&idx)[kStageSize], Idx(&ridx)[kStageSize],
                                            bst_bin_t(&gidx)[kStageSize],
                                            GradientPair(&gpair)[kStageSize], int stage) {
#pragma unroll
      for (int i = 0; i < kStageSize; i++) {
        // Avoid atomic add if it's a null value.
        auto fidx = FeatIdx(group_, idx[i], feature_stride_);
        auto itidx = IterIdx(matrix_, ridx[i], fidx);
        auto shmem_beg_idx = kBlockThreads * i * kBufSize + (threadIdx.x * kBufSize);
        shmem_beg_idx = shmem_beg_idx + stage * kBlockThreads * kStageSize * kBufSize;

        bst_bin_t ngidx = matrix_.gidx_iter.ReadBuf(itidx, stage_buf + shmem_beg_idx);
        bst_bin_t kidx = matrix_.gidx_iter[itidx];
        SPAN_CHECK(kidx == ngidx);
        if (kDense || ngidx != matrix_.NullValue()) {
          if constexpr (kCompressed) {
            ngidx += matrix_.feature_segments[fidx];
          }
        } else {
          // Use -1 to denote missing. Since we need to add the beginning bin to gidx, the
          // result might equal to the `NullValue`.
          ngidx = -1;
          SPAN_CHECK(false);
        }
        SPAN_CHECK(ngidx == gidx[i]);
        // kDense || gidx[i] != -1
        if (kDense || ngidx != -1) {
          auto adjusted = rounding_.ToFixedPoint(gpair[i]);

          // AtomicAddGpairShared
          fn(gidx[i] - group_.start_bin, adjusted);
        }
      }
    };

    auto stage = 0;

    auto flip_stage = [&] {
      stage = (stage + 1) % kStages;
    };

    pipe.producer_acquire();
    if (offset + kItemsPerTile <= n_elements_) {
      load(idx_s[stage], ridx_s[stage], gpair_s[stage], stage, offset);
      load_gidx(idx_s[stage], ridx_s[stage], gidx_s[stage], stage);
    }
    pipe.producer_commit();

    flip_stage();  // s -> 1

    pipe.producer_acquire();
    if (offset + kItemsPerTile <= n_elements_) {
      load(idx_s[stage], ridx_s[stage], gpair_s[stage], stage, offset);
      load_gidx(idx_s[stage], ridx_s[stage], gidx_s[stage], stage);
    }
    pipe.producer_commit();

    flip_stage();  // s -> 0

    while (offset + kItemsPerTile <= n_elements_) {
      if (threadIdx.x == 0) {
        printf("off: %d\n", int(offset));
      }
      // Consume
      cuda::pipeline_consumer_wait_prior<1>(pipe);
      write_gidx(idx_s[stage], ridx_s[stage], gidx_s[stage], gpair_s[stage], stage);
      pipe.consumer_release();

      // Re-fill
      pipe.producer_acquire();
      offset += (kItemsPerTile * gridDim.x) * ((stage + 1) % 2);
      if (offset + kItemsPerTile <= n_elements_) {
        load(idx_s[stage], ridx_s[stage], gpair_s[stage], stage, offset);
        load_gidx(idx_s[stage], ridx_s[stage], gidx_s[stage], stage);
      }
      flip_stage();
      pipe.producer_commit();
    }

    ProcessPartialTileShared(offset, fn);

    // Write shared memory back to global memory
    __syncthreads();
    for (auto i : dh::BlockStrideRange(0, group_.num_bins)) {
      gfn(d_node_hist_ + group_.start_bin + i, smem_arr_[i]);
    }
  }

  template <typename Gfn>
  __device__ void BuildHistogramWithGlobal(Gfn&& gfn) {
    for (auto idx : dh::GridStrideRange(static_cast<std::size_t>(0), n_elements_)) {
      Idx ridx = d_ridx_[idx / feature_stride_];
      auto fidx = FeatIdx(group_, idx, feature_stride_);
      bst_bin_t compressed_bin = matrix_.gidx_iter[IterIdx(matrix_, ridx, fidx)];
      if (compressed_bin != matrix_.NullValue()) {
        if (kCompressed) {
          compressed_bin += this->matrix_.feature_segments[fidx];
        }
        auto adjusted = rounding_.ToFixedPoint(d_gpair_[ridx]);
        gfn(d_node_hist_ + compressed_bin, adjusted);
      }
    }
  }
};
}  // namespace xgboost::tree::cuda_impl
