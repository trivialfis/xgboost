/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#include <thrust/iterator/transform_iterator.h>  // for make_transform_iterator

#include <algorithm>
#include <cstdint>  // uint32_t, int32_t
#include <cuda/barrier>

#include "../../collective/aggregator.h"
#include "../../common/deterministic.cuh"
#include "../../common/device_helpers.cuh"
#include "../../common/linalg_op.cuh"  // for tbegin
#include "../../common/nvtx_utils.h"   // for xgboost_NVTX_FN_RANGE
#include "../../data/ellpack_page.cuh"
#include "histogram.cuh"
#include "row_partitioner.cuh"
#include "xgboost/base.h"

namespace xgboost::tree {
namespace {
struct Pair {
  GradientPair first;
  GradientPair second;
};
__host__ XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}

XGBOOST_DEV_INLINE bst_feature_t FeatIdx(FeatureGroup const& group, bst_idx_t idx,
                                         std::int32_t feature_stride) {
  auto fidx = group.start_feature + idx % feature_stride;
  return fidx;
}

template <typename IterT>
XGBOOST_DEV_INLINE bst_idx_t IterIdx(EllpackAccessorImpl<IterT> const& matrix,
                                     RowPartitioner::RowIndexT ridx, bst_feature_t fidx) {
  // ridx_local = ridx - base_rowid  <== Row index local to each batch
  // entry_idx = ridx_local * row_stride <== Starting entry index for this row in the matrix
  // entry_idx += start_feature  <== Inside a row, first column inside this feature group
  // idx % feature_stride <== The feaature index local to the current feature group
  // entry_idx += idx % feature_stride <== Final index.
  return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
}
}  // anonymous namespace

struct Clip {
  static XGBOOST_DEV_INLINE float Pclip(float v) { return v > 0 ? v : 0; }
  static XGBOOST_DEV_INLINE float Nclip(float v) { return v < 0 ? abs(v) : 0; }

  XGBOOST_DEV_INLINE Pair operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return {GradientPair{pg, ph}, GradientPair{ng, nh}};
  }
};

/**
 * In algorithm 5 (see common::CreateRoundingFactor) the bound is calculated as
 * $max(|v_i|) * n$.  Here we use the bound:
 *
 * \begin{equation}
 *   max( fl(\sum^{V}_{v_i>0}{v_i}), fl(\sum^{V}_{v_i<0}|v_i|) )
 * \end{equation}
 *
 * to avoid outliers, as the full reduction is reproducible on GPU with reduction tree.
 */
GradientQuantiser::GradientQuantiser(Context const* ctx, common::Span<GradientPair const> gpair,
                                     MetaInfo const& info) {
  using GradientSumT = GradientPairPrecise;
  using T = typename GradientSumT::ValueT;

  thrust::device_ptr<GradientPair const> gpair_beg{gpair.data()};
  auto beg = thrust::make_transform_iterator(gpair_beg, Clip());
  Pair p = dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + gpair.size(), Pair{}, thrust::plus<Pair>{});
  // Treat pair as array of 4 primitive types to allreduce
  using ReduceT = typename decltype(p.first)::ValueT;
  static_assert(sizeof(Pair) == sizeof(ReduceT) * 4, "Expected to reduce four elements.");
  auto rc = collective::GlobalSum(ctx, info, linalg::MakeVec(reinterpret_cast<ReduceT*>(&p), 4));
  collective::SafeColl(rc);

  GradientPair positive_sum{p.first}, negative_sum{p.second};

  std::size_t total_rows = gpair.size();
  rc = collective::GlobalSum(ctx, info, linalg::MakeVec(&total_rows, 1));
  collective::SafeColl(rc);

  auto histogram_rounding =
      GradientSumT{common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetGrad(), negative_sum.GetGrad()), total_rows),
                   common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetHess(), negative_sum.GetHess()), total_rows)};

  using IntT = typename GradientPairInt64::ValueT;

  /**
   * Factor for converting gradients from fixed-point to floating-point.
   */
  to_floating_point_ =
      histogram_rounding /
      static_cast<T>(static_cast<IntT>(1)
                     << (sizeof(typename GradientSumT::ValueT) * 8 - 2));  // keep 1 for sign bit
  /**
   * Factor for converting gradients from floating-point to fixed-point. For
   * f64:
   *
   *   Precision = 64 - 1 - log2(rounding)
   *
   * rounding is calcuated as exp(m), see the rounding factor calcuation for
   * details.
   */
  to_fixed_point_ = GradientSumT(static_cast<T>(1) / to_floating_point_.GetGrad(),
                                 static_cast<T>(1) / to_floating_point_.GetHess());
}

MultiGradientQuantiser::MultiGradientQuantiser(Context const* ctx,
                                               linalg::MatrixView<GradientPair const> gpair,
                                               MetaInfo const& info) {
  CHECK(gpair.FContiguous());
  std::vector<GradientQuantiser> h_quantizers;
  // TODO(jiamingy): We need to merge this into a single call for improved distributed training.
  for (bst_target_t t = 0, n_targets = gpair.Shape(1); t < n_targets; ++t) {
    h_quantizers.emplace_back(ctx, gpair.Slice(linalg::All(), t).Values(), info);
  }
  this->quantizers_ = h_quantizers;
}

namespace cuda_impl {
void TransposeGradient(Context const* ctx, linalg::MatrixView<GradientPair const> in,
                       linalg::MatrixView<GradientPair> out) {
  CHECK(in.CContiguous());
  CHECK(out.FContiguous());
  thrust::copy_n(ctx->CUDACtx()->CTP(), in.Values().data(), in.Size(), linalg::tbegin(out));
}
}  // namespace cuda_impl

XGBOOST_DEV_INLINE void AtomicAddGpairShared(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<int64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  AtomicAdd64As32(dst_ptr, g);
  AtomicAdd64As32(dst_ptr + 1, h);
}

// Global 64 bit integer atomics at the time of writing do not benefit from being separated into two
// 32 bit atomics
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

  __device__ void ProcessPartialTileShared(std::size_t offset) {
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
        AtomicAddGpairShared(smem_arr_ + compressed_bin - group_.start_bin, adjusted);
      }
    }
  }

  // Instruction level parallelism by loop unrolling
  // Allows the kernel to pipeline many operations while waiting for global memory
  // Increases the throughput of this kernel significantly
  __device__ void ProcessFullTileShared(std::size_t offset) {
    std::size_t idx[kItemsPerThread];
    Idx ridx[kItemsPerThread];
    bst_bin_t gidx[kItemsPerThread];
    GradientPair gpair[kItemsPerThread];
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      idx[i] = offset + i * kBlockThreads + threadIdx.x;
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      ridx[i] = d_ridx_[idx[i] / feature_stride_];
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      gpair[i] = d_gpair_[ridx[i]];
      auto fidx = FeatIdx(group_, idx[i], feature_stride_);
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
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      // Avoid atomic add if it's a null value.
      if (kDense || gidx[i] != -1) {
        auto adjusted = rounding_.ToFixedPoint(gpair[i]);
        AtomicAddGpairShared(smem_arr_ + gidx[i] - group_.start_bin, adjusted);
      }
    }
  }
  __device__ void BuildHistogramWithShared() {
    dh::BlockFill(smem_arr_, group_.num_bins, GradientPairInt64{});
    __syncthreads();

    std::size_t offset = blockIdx.x * kItemsPerTile;
    while (offset + kItemsPerTile <= n_elements_) {
      ProcessFullTileShared(offset);
      offset += kItemsPerTile * gridDim.x;
    }
    ProcessPartialTileShared(offset);

    // Write shared memory back to global memory
    __syncthreads();
    for (auto i : dh::BlockStrideRange(0, group_.num_bins)) {
      AtomicAddGpairGlobal(d_node_hist_ + group_.start_bin + i, smem_arr_[i]);
    }
  }

  __device__ void BuildHistogramWithGlobal() {
    for (auto idx : dh::GridStrideRange(static_cast<std::size_t>(0), n_elements_)) {
      Idx ridx = d_ridx_[idx / feature_stride_];
      auto fidx = FeatIdx(group_, idx, feature_stride_);
      bst_bin_t compressed_bin = matrix_.gidx_iter[IterIdx(matrix_, ridx, fidx)];
      if (compressed_bin != matrix_.NullValue()) {
        if (kCompressed) {
          compressed_bin += this->matrix_.feature_segments[fidx];
        }
        auto adjusted = rounding_.ToFixedPoint(d_gpair_[ridx]);
        AtomicAddGpairGlobal(d_node_hist_ + compressed_bin, adjusted);
      }
    }
  }
};

template <typename Accessor, bool kCompressed, bool kDense, bool use_shared_memory_histograms,
          int kBlockThreads, int kItemsPerThread>
__global__ void __launch_bounds__(kBlockThreads)
    SharedMemHistKernel(Accessor const matrix, const FeatureGroupsAccessor feature_groups,
                        common::Span<const RowPartitioner::RowIndexT> d_ridx,
                        GradientPairInt64* __restrict__ d_node_hist,
                        const GradientPair* __restrict__ d_gpair,
                        GradientQuantiser const rounding) {
  extern __shared__ char smem[];
  const FeatureGroup group = feature_groups[blockIdx.y];
  auto smem_arr = reinterpret_cast<GradientPairInt64*>(smem);
  auto agent = HistogramAgent<Accessor, kCompressed, kDense, kBlockThreads, kItemsPerThread>(
      smem_arr, d_node_hist, group, matrix, d_ridx, rounding, d_gpair);
  if (use_shared_memory_histograms) {
    agent.BuildHistogramWithShared();
  } else {
    agent.BuildHistogramWithGlobal();
  }
}

// Kernel for vector-leaf, bare minimum for now.
template <typename Accessor, bool kCompressed, bool kDense, bool use_shared_memory_histograms,
          std::int32_t kBlockThreads, std::int32_t kItemsPerThread>
__global__ __launch_bounds__(kBlockThreads) void MultiHistKernel(
    Accessor const matrix, const FeatureGroupsAccessor feature_groups,
    common::Span<const RowPartitioner::RowIndexT> d_ridx, GradientPairInt64* d_node_hist,
    linalg::MatrixView<const GradientPair> d_gpair,
    common::Span<GradientQuantiser const> roundings) {
  const FeatureGroup group = feature_groups[blockIdx.y];
  std::int32_t feature_stride = kCompressed ? group.num_features : matrix.row_stride;
  bst_idx_t n_elements = feature_stride * d_ridx.size();
  using Idx = RowPartitioner::RowIndexT;
  for (auto idx : dh::GridStrideRange(static_cast<std::size_t>(0), n_elements)) {
    Idx ridx = d_ridx[idx / feature_stride];
    auto fidx = FeatIdx(group, idx, feature_stride);
    bst_bin_t compressed_bin = matrix.gidx_iter[IterIdx(matrix, ridx, fidx)];
    if (compressed_bin != matrix.NullValue()) {
      if (kCompressed) {
        compressed_bin += matrix.feature_segments[fidx];
      }
      bst_target_t n_targets = roundings.size();
      compressed_bin *= n_targets;
      // TODO(jiamingy): Assign a thread for each target.
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto adjusted = roundings[t].ToFixedPoint(d_gpair(ridx, t));
        AtomicAddGpairGlobal(d_node_hist + compressed_bin + t, adjusted);
      }
    }
  }
}

namespace {
constexpr std::int32_t kBlockThreads = 1024;
constexpr std::int32_t kItemsPerThread = 8;
constexpr std::int32_t ItemsPerTile() { return kBlockThreads * kItemsPerThread; }
template <auto Ker>
using DeduceKernelT = std::decay_t<decltype(Ker)>;
}  // namespace

// Use auto deduction guide to workaround compiler error.
template <typename Accessor>
struct HistogramKernel {
  enum KernelType : std::size_t {
    // single-target
    kGlobalCompr = 0,
    kGlobal = 1,
    kSharedCompr = 2,
    kShared = 3,
    kGlobalDense = 4,
    kSharedDense = 5,
    // multi-target
    kMtGlobalCompr = 6,
    kMtGlobal = 7,
    kMtSharedCompr = 8,
    kMtShared = 9,
    kMtGlobalDense = 10,
    kMtSharedDense = 11,
  };
  /**
   * Single-target
   */
  // Kernel for working with compressed sparse Ellpack using the global memory.
  using GlobalCompr = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, false, false, kBlockThreads, kItemsPerThread>>;
  GlobalCompr global_compr_kernel{
      SharedMemHistKernel<Accessor, true, false, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the global memory.
  using Global = DeduceKernelT<
      SharedMemHistKernel<Accessor, false, false, false, kBlockThreads, kItemsPerThread>>;
  Global global_kernel{
      SharedMemHistKernel<Accessor, false, false, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed sparse Ellpack using the shared memory.
  using SharedCompr = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, false, true, kBlockThreads, kItemsPerThread>>;
  SharedCompr shared_compr_kernel{
      SharedMemHistKernel<Accessor, true, false, true, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the shared memory.
  using Shared = DeduceKernelT<
      SharedMemHistKernel<Accessor, false, false, true, kBlockThreads, kItemsPerThread>>;
  Shared shared_kernel{
      SharedMemHistKernel<Accessor, false, false, true, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed dense ellpack using the global memory
  using GlobalDense = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, true, false, kBlockThreads, kItemsPerThread>>;
  GlobalDense global_dense_kernel{
      SharedMemHistKernel<Accessor, true, true, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed dense ellpack using the shared memory
  using SharedDense = DeduceKernelT<
      SharedMemHistKernel<Accessor, true, true, true, kBlockThreads, kItemsPerThread>>;
  SharedDense shared_dense_kernel{
      SharedMemHistKernel<Accessor, true, true, true, kBlockThreads, kItemsPerThread>};

  /**
   * Multi-target
   */
  // Kernel for working with compressed sparse Ellpack using the global memory.
  using MtGlobalCompr =
      DeduceKernelT<MultiHistKernel<Accessor, true, false, false, kBlockThreads, kItemsPerThread>>;
  MtGlobalCompr mt_global_compr_kernel{
      MultiHistKernel<Accessor, true, false, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the global memory.
  using MtGlobal =
      DeduceKernelT<MultiHistKernel<Accessor, false, false, false, kBlockThreads, kItemsPerThread>>;
  MtGlobal mt_global_kernel{
      MultiHistKernel<Accessor, false, false, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed sparse Ellpack using the shared memory.
  using MtSharedCompr =
      DeduceKernelT<MultiHistKernel<Accessor, true, false, true, kBlockThreads, kItemsPerThread>>;
  MtSharedCompr mt_shared_compr_kernel{
      MultiHistKernel<Accessor, true, false, true, kBlockThreads, kItemsPerThread>};
  // Kernel for working with sparse Ellpack using the shared memory.
  using MtShared =
      DeduceKernelT<MultiHistKernel<Accessor, false, false, true, kBlockThreads, kItemsPerThread>>;
  MtShared mt_shared_kernel{
      MultiHistKernel<Accessor, false, false, true, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed dense ellpack using the global memory
  using MtGlobalDense =
      DeduceKernelT<MultiHistKernel<Accessor, true, true, false, kBlockThreads, kItemsPerThread>>;
  MtGlobalDense mt_global_dense_kernel{
      MultiHistKernel<Accessor, true, true, false, kBlockThreads, kItemsPerThread>};
  // Kernel for working with compressed dense ellpack using the shared memory
  using MtSharedDense =
      DeduceKernelT<MultiHistKernel<Accessor, true, true, true, kBlockThreads, kItemsPerThread>>;
  MtSharedDense mt_shared_dense_kernel{
      MultiHistKernel<Accessor, true, true, true, kBlockThreads, kItemsPerThread>};

  bool shared{false};
  std::array<std::uint32_t, 12> grid_sizes;
  std::size_t smem_size{0};
  std::size_t const max_shared_memory;
  bool const force_global;

  HistogramKernel(Context const* ctx, FeatureGroupsAccessor const& feature_groups,
                  bool force_global_memory)
      : max_shared_memory{dh::MaxSharedMemoryOptin(ctx->Ordinal())},
        force_global{force_global_memory} {
    std::fill_n(grid_sizes.data(), grid_sizes.size(), 0);
    // Decide whether to use shared memory
    // Opt into maximum shared memory for the kernel if necessary
    this->smem_size = feature_groups.ShmemSize();
    this->shared = !force_global_memory && this->smem_size <= this->max_shared_memory;
    this->smem_size = this->shared ? this->smem_size : 0;

    auto init = [&](auto& kernel, KernelType k) {
      if (this->shared) {
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           this->max_shared_memory));
      }

      // determine the launch configuration
      std::int32_t num_groups = feature_groups.NumGroups();
      std::int32_t n_mps = 0;
      dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, ctx->Ordinal()));

      std::int32_t n_blocks_per_mp = 0;
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks_per_mp, kernel,
                                                                  kBlockThreads, this->smem_size));

      // This gives the number of blocks to keep the device occupied Use this as the
      // maximum number of blocks
      this->grid_sizes[static_cast<std::size_t>(k)] = n_blocks_per_mp * n_mps;
    };
    // Initialize all kernel instantiations
    {
      // Single target
      std::array kernel_types{kGlobalCompr, kGlobal,      kSharedCompr,
                              kShared,      kGlobalDense, kSharedDense};
      std::int32_t k = 0;
      for (auto& kernel : {global_compr_kernel, global_kernel, shared_compr_kernel, shared_kernel,
                           global_dense_kernel, shared_dense_kernel}) {
        init(kernel, kernel_types[k]);
        ++k;
      }
    }
    {
      // Multi target
      std::array kernel_types{kMtGlobalCompr, kMtGlobal,      kMtSharedCompr,
                              kMtShared,      kMtGlobalDense, kMtSharedDense};
      std::int32_t k = 0;
      for (auto& kernel : {mt_global_compr_kernel, mt_global_kernel, mt_shared_compr_kernel,
                           mt_shared_kernel, mt_global_dense_kernel, mt_shared_dense_kernel}) {
        init(kernel, kernel_types[k]);
        ++k;
      }
    }
  }
};

template <typename Accessor>
class DeviceHistogramDispatchAccessor {
  std::unique_ptr<HistogramKernel<Accessor>> kernel_{nullptr};

 public:
  void Reset(Context const* ctx, FeatureGroupsAccessor const& feature_groups,
             bool force_global_memory) {
    this->kernel_ =
        std::make_unique<HistogramKernel<Accessor>>(ctx, feature_groups, force_global_memory);
    if (force_global_memory) {
      CHECK(!this->kernel_->shared);
    }
  }

  void BuildHistogram(CUDAContext const* ctx, Accessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      common::Span<GradientPair const> gpair,
                      common::Span<const cuda_impl::RowIndexT> d_ridx,
                      common::Span<GradientPairInt64> histogram, GradientQuantiser rounding) const {
    CHECK(kernel_);
    // Otherwise launch blocks such that each block has a minimum amount of work to do
    // There are fixed costs to launching each block, e.g. zeroing shared memory
    // The below amount of minimum work was found by experimentation
    int columns_per_group = common::DivRoundUp(matrix.row_stride, feature_groups.NumGroups());
    // Average number of matrix elements processed by each group
    std::size_t items_per_group = d_ridx.size() * columns_per_group;

    // Allocate number of blocks such that each block has about kMinItemsPerBlock work
    // Up to a maximum where the device is saturated
    auto constexpr kMinItemsPerBlock = ItemsPerTile();

    auto launcher = [&](auto const& kernel, std::uint32_t grid_size) {
      CHECK_NE(grid_size, 0);
      grid_size = std::min(grid_size, static_cast<std::uint32_t>(
                                          common::DivRoundUp(items_per_group, kMinItemsPerBlock)));
      dh::LaunchKernel{dim3(grid_size, feature_groups.NumGroups()),  // NOLINT
                       static_cast<uint32_t>(kBlockThreads), kernel_->smem_size, ctx->Stream()}(
          kernel, matrix, feature_groups, d_ridx, histogram.data(), gpair.data(), rounding);
    };

    using K = HistogramKernel<EllpackDeviceAccessor>::KernelType;
    if (!this->kernel_->shared) {  // Use global memory
      CHECK_EQ(this->kernel_->smem_size, 0);
      if (matrix.IsDense()) {
        CHECK(this->kernel_->force_global ||
              (feature_groups.ShmemSize() >= this->kernel_->max_shared_memory));
        launcher(this->kernel_->global_dense_kernel, this->kernel_->grid_sizes[K::kGlobalDense]);
      } else if (matrix.IsDenseCompressed()) {
        CHECK(this->kernel_->force_global ||
              (feature_groups.ShmemSize() >= this->kernel_->max_shared_memory));
        launcher(this->kernel_->global_compr_kernel, this->kernel_->grid_sizes[K::kGlobalCompr]);
      } else {
        // Sparse
        launcher(this->kernel_->global_kernel, this->kernel_->grid_sizes[K::kGlobal]);
      }
    } else {  // Use shared memory
      CHECK_NE(this->kernel_->smem_size, 0);
      if (matrix.IsDense()) {
        launcher(this->kernel_->shared_dense_kernel, this->kernel_->grid_sizes[K::kSharedDense]);
      } else if (matrix.IsDenseCompressed()) {
        // Dense
        launcher(this->kernel_->shared_compr_kernel, this->kernel_->grid_sizes[K::kSharedCompr]);
      } else {
        // Sparse
        launcher(this->kernel_->shared_kernel, this->kernel_->grid_sizes[K::kShared]);
      }
    }
  }

  void BuildHistogram(CUDAContext const* ctx, Accessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      linalg::MatrixView<GradientPair const> gpair,
                      common::Span<const cuda_impl::RowIndexT> d_ridx,
                      common::Span<GradientPairInt64> histogram,
                      common::Span<GradientQuantiser const> roundings) const {
    CHECK(kernel_);
    // Otherwise launch blocks such that each block has a minimum amount of work to do
    // There are fixed costs to launching each block, e.g. zeroing shared memory
    // The below amount of minimum work was found by experimentation
    int columns_per_group = common::DivRoundUp(matrix.row_stride, feature_groups.NumGroups());
    // Average number of matrix elements processed by each group
    std::size_t items_per_group = d_ridx.size() * columns_per_group;

    // Allocate number of blocks such that each block has about kMinItemsPerBlock work
    // Up to a maximum where the device is saturated
    auto constexpr kMinItemsPerBlock = ItemsPerTile();

    auto launcher = [&](auto const& kernel, std::uint32_t grid_size) {
      CHECK_NE(grid_size, 0);
      grid_size = std::min(grid_size, static_cast<std::uint32_t>(
                                          common::DivRoundUp(items_per_group, kMinItemsPerBlock)));
      dh::LaunchKernel{dim3(grid_size, feature_groups.NumGroups()),  // NOLINT
                       static_cast<uint32_t>(kBlockThreads), kernel_->smem_size, ctx->Stream()}(
          kernel, matrix, feature_groups, d_ridx, histogram.data(), gpair, roundings);
    };

    using K = HistogramKernel<EllpackDeviceAccessor>::KernelType;
    if (!this->kernel_->shared) {  // Use global memory
      CHECK_EQ(this->kernel_->smem_size, 0);
      if (matrix.IsDense()) {
        CHECK(this->kernel_->force_global ||
              (feature_groups.ShmemSize() >= this->kernel_->max_shared_memory));
        launcher(this->kernel_->mt_global_dense_kernel,
                 this->kernel_->grid_sizes[K::kMtGlobalDense]);
      } else if (matrix.IsDenseCompressed()) {
        CHECK(this->kernel_->force_global ||
              (feature_groups.ShmemSize() >= this->kernel_->max_shared_memory));
        launcher(this->kernel_->mt_global_compr_kernel,
                 this->kernel_->grid_sizes[K::kMtGlobalCompr]);
      } else {
        // Sparse
        launcher(this->kernel_->mt_global_kernel, this->kernel_->grid_sizes[K::kMtGlobal]);
      }
    } else {  // Use shared memory
      CHECK_NE(this->kernel_->smem_size, 0);
      CHECK(false) << MTNotImplemented();
      if (matrix.IsDense()) {
        launcher(this->kernel_->mt_shared_dense_kernel,
                 this->kernel_->grid_sizes[K::kMtSharedDense]);
      } else if (matrix.IsDenseCompressed()) {
        // Dense
        launcher(this->kernel_->mt_shared_compr_kernel,
                 this->kernel_->grid_sizes[K::kMtSharedCompr]);
      } else {
        // Sparse
        launcher(this->kernel_->mt_shared_kernel, this->kernel_->grid_sizes[K::kMtShared]);
      }
    }
  }
};

// Dispatch between single buffer accessor and double buffer accessor.
struct DeviceHistogramBuilderImpl {
  DeviceHistogramDispatchAccessor<EllpackDeviceAccessor> simpl;
  DeviceHistogramDispatchAccessor<DoubleEllpackAccessor> dimpl;

  template <typename... Args>
  void Reset(Args&&... args) {
    this->simpl.Reset(std::forward<Args>(args)...);
    this->dimpl.Reset(std::forward<Args>(args)...);
  }

  template <typename Accessor, typename... Args>
  void BuildHistogram(CUDAContext const* ctx, Accessor const& matrix, Args&&... args) {
    if constexpr (std::is_same_v<Accessor, EllpackDeviceAccessor>) {
      this->simpl.BuildHistogram(ctx, matrix, std::forward<Args>(args)...);
    } else {
      static_assert(std::is_same_v<Accessor, DoubleEllpackAccessor>);
      this->dimpl.BuildHistogram(ctx, matrix, std::forward<Args>(args)...);
    }
  }
};

DeviceHistogramBuilder::DeviceHistogramBuilder()
    : p_impl_{std::make_unique<DeviceHistogramBuilderImpl>()} {
  monitor_.Init(__func__);
}

DeviceHistogramBuilder::~DeviceHistogramBuilder() = default;

void DeviceHistogramBuilder::Reset(Context const* ctx, std::size_t max_cached_hist_nodes,
                                   FeatureGroupsAccessor const& feature_groups,
                                   bst_bin_t n_total_bins, bool force_global_memory) {
  this->monitor_.Start(__func__);
  this->p_impl_->Reset(ctx, feature_groups, force_global_memory);
  this->hist_.Reset(ctx, n_total_bins, max_cached_hist_nodes);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(CUDAContext const* ctx, EllpackAccessor const& matrix,
                                            FeatureGroupsAccessor const& feature_groups,
                                            common::Span<GradientPair const> gpair,
                                            common::Span<const cuda_impl::RowIndexT> ridx,
                                            common::Span<GradientPairInt64> histogram,
                                            GradientQuantiser rounding) {
  this->monitor_.Start(__func__);
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridx, histogram,
                                      rounding);
      },
      matrix);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(CUDAContext const* ctx, EllpackAccessor const& matrix,
                                            FeatureGroupsAccessor const& feature_groups,
                                            linalg::MatrixView<GradientPair const> gpair,
                                            common::Span<const std::uint32_t> ridx,
                                            common::Span<GradientPairInt64> histogram,
                                            common::Span<GradientQuantiser const> roundings) {
  xgboost_NVTX_FN_RANGE();
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridx, histogram,
                                      roundings);
      },
      matrix);
}

template <std::int32_t kBlockThreadsIn, std::int32_t kItemsPerThreadIn, bool kCompressedIn>
struct HistPolicy {
  static constexpr std::int32_t kBlockThreads = kBlockThreadsIn;
  static constexpr std::int32_t kItemsPerThread = kItemsPerThreadIn;
  static constexpr std::int32_t kTileSize = kBlockThreadsIn * kItemsPerThreadIn;
  static constexpr bool kCompressed = kCompressedIn;
};

template <typename Policy, typename Accessor, typename RidxIterSpan>
__global__ __launch_bounds__(Policy::kBlockThreads) void HistKernel(
    Accessor const matrix, FeatureGroupsAccessor const feature_groups, RidxIterSpan* d_ridx_iters,
    common::Span<GradientPairInt64>* node_hists, bst_node_t n_nodes,
    linalg::MatrixView<GradientPair const> d_gpair,
    common::Span<GradientQuantiser const> roundings) {
  auto d_roundings = roundings.data();
  auto nidx_in_set = blockIdx.z;
  FeatureGroup group = feature_groups[blockIdx.y];
  auto d_ridx = d_ridx_iters[nidx_in_set];
  std::int32_t feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;

  // grid stride loop
  auto const kStride = Policy::kTileSize * gridDim.x;
  // first grid
  std::size_t offset = blockIdx.x * Policy::kTileSize;

  bst_idx_t n_elements = feature_stride * d_ridx.size();

  auto prefetch_gidx_tile = [&](auto idx, auto ridx) {
    if (__shfl_up_sync(0xFFFFFFFF, ridx, 1) != ridx) {
      auto fidx = FeatIdx(group, idx, feature_stride);
      matrix.gidx_iter.Prefetch(IterIdx(matrix, ridx, fidx));
    }
  };

  // {
  //   std::int32_t const valid_items =
  //       cuda::std::min(n_elements - offset, static_cast<std::size_t>(Policy::kTileSize));
  //   if (Policy::kTileSize == valid_items) {
  //     for (int j = 0; j < Policy::kItemsPerThread; ++j) {
  //       const int idx = offset + j * Policy::kBlockThreads + threadIdx.x;
  //       prefetch_gidx_tile(idx, valid_items);
  //     }
  //   }
  // }

  using Idx = RowPartitioner::RowIndexT;
  bst_target_t const n_targets = roundings.size();

  extern __align__(cuda::std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];
  auto node_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  dh::BlockFill(node_hist, group.num_bins, GradientPairInt64{});

  auto d_node_hist = node_hists[nidx_in_set].data();

  __syncthreads();

  auto prefetch_gpair_tile = [&](auto idx, auto ridx) {
    common::PrefetchGlobalL2(&d_gpair(ridx, 0));
  };

  auto process_valid_tile = [&](auto idx) {
    Idx ridx = d_ridx[idx / feature_stride];
    auto fidx = FeatIdx(group, idx, feature_stride);
    bst_bin_t compressed_bin = matrix.gidx_iter[IterIdx(matrix, ridx, fidx)];
    if (compressed_bin != matrix.NullValue()) {
      if (Policy::kCompressed) {
        compressed_bin += matrix.feature_segments[fidx];
      }
      compressed_bin *= n_targets;  // fixme (group.start_bin)
      // TODO(jiamingy): Assign a thread for each target.
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto adjusted = d_roundings[t].ToFixedPoint(d_gpair(ridx, t));
        AtomicAddGpairShared(node_hist + compressed_bin - group.start_bin, adjusted);
      }
    }
  };

  auto process_gpair_tile = [&](auto full_tile, auto offset, auto valid_items) {
    for (int j = 0; j < Policy::kItemsPerThread; ++j) {
      if (full_tile) {
        const int idx = offset + j * Policy::kBlockThreads + threadIdx.x;
        Idx ridx = d_ridx[idx / feature_stride];
        prefetch_gpair_tile(idx, ridx);
        prefetch_gidx_tile(idx, ridx);
      }
    }
    for (int j = 0; j < Policy::kItemsPerThread; ++j) {
      const int idx = offset + j * Policy::kBlockThreads + threadIdx.x;
      if (full_tile || idx < valid_items) {
        // if (j != Policy::kItemsPerThread - 1) {
        //   const int idx = offset + (j + 1) * Policy::kBlockThreads + threadIdx.x;
        //   Idx ridx = d_ridx[idx / feature_stride];
        //   prefetch_gpair_tile(idx, ridx);
        //   prefetch_gidx_tile(idx, ridx);
        // }
        process_valid_tile(idx);
      }
    }
  };

  while (offset < n_elements) {
    std::int32_t const valid_items =
        cuda::std::min(n_elements - offset, static_cast<std::size_t>(Policy::kTileSize));
    if (Policy::kTileSize == valid_items) {
      process_gpair_tile(std::true_type{}, offset, valid_items);
    } else {
      process_gpair_tile(std::false_type{}, offset, valid_items);
    }
    offset += kStride;
  }

  // Write shared memory back to global memory
  __syncthreads();
  for (auto i : dh::BlockStrideRange(0, group.num_bins)) {
    // fixme: n targets, need to handle it in the feature groups as well.
    if (node_hist[i].GetQuantisedHess() == -1) {
      AtomicAddGpairGlobal(d_node_hist + group.start_bin + i, node_hist[i]);
    }
  }
}

void DeviceHistogramBuilder::BuildHistogram(
    CUDAContext const* ctx, EllpackAccessor const& matrix,
    FeatureGroupsAccessor const& feature_groups, linalg::MatrixView<GradientPair const> gpair,
    common::Span<common::Span<cuda_impl::RowIndexT const>> ridxs,
    common::Span<common::Span<GradientPairInt64>> hists, std::size_t n_max_samples,
    common::Span<GradientQuantiser const> roundings) {
  CHECK_EQ(ridxs.size(), hists.size());
  auto n_nodes = hists.size();

  constexpr int kBlockThreads = 1024;
  constexpr int kItemsPerThread = 4;
  auto launch = [&](auto policy, auto kernel, auto acc, auto ridx_iters) {
    // fixme: support global-only.
    using Policy = common::GetValueT<decltype(policy)>;

    int columns_per_group = common::DivRoundUp(acc.row_stride, feature_groups.NumGroups());
    // Average number of matrix elements processed by each group
    std::size_t items_per_group = n_max_samples * columns_per_group;

    auto n_grids = common::DivRoundUp(items_per_group, Policy::kTileSize);
    CHECK_GT(n_grids, 0);

    std::int32_t num_groups = feature_groups.NumGroups();
    std::int32_t n_mps = 0;
    dh::safe_cuda(
        cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, 0));  // fixme, ordinal

    std::int32_t n_blocks_per_mp = 0;
    auto shmem_bytes = feature_groups.ShmemSize();

    // fixme: blocking call.
    dh::safe_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));

    dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n_blocks_per_mp, kernel, Policy::kBlockThreads, shmem_bytes));
    CHECK_GE(n_blocks_per_mp, 1);

    n_grids = std::min(n_blocks_per_mp * n_mps, static_cast<std::int32_t>(n_grids));

    CHECK_GE(roundings.size(), 1);
    CHECK_GE(feature_groups.NumGroups(), 1);
    dim3 conf(n_grids, feature_groups.NumGroups(), n_nodes);
    std::cout << "x:" << conf.x << " y:" << conf.y << " z:" << conf.z << " n_grids:" << n_grids
              << ",columns_per_group: " << columns_per_group << std::endl;
    kernel<<<conf, Policy::kBlockThreads, shmem_bytes, ctx->Stream()>>>(
        acc, feature_groups, ridx_iters, hists.data(), hists.size(), gpair, roundings);
    dh::safe_cuda(cudaPeekAtLastError());
  };

  std::visit(
      [&](auto&& acc) {
        using AccessorT = common::GetValueT<decltype(acc)>;
        using Policy = HistPolicy<kBlockThreads, kItemsPerThread, true>;

        if (ridxs.size() == 1 && n_max_samples == acc.n_rows) {
          using RidxIter = thrust::counting_iterator<cuda_impl::RowIndexT>;
          dh::caching_device_vector<common::IterSpan<RidxIter>> ridx_iters(
              hists.size(), common::IterSpan{thrust::make_counting_iterator(0u), gpair.Shape(0)});
          auto kernel = HistKernel<Policy, AccessorT, common::IterSpan<RidxIter>>;
          launch(Policy{}, kernel, acc, ridx_iters.data().get());
        } else {
          using RidxIter = cuda_impl::RowIndexT const;
          auto kernel = HistKernel<Policy, AccessorT, common::Span<RidxIter>>;
          launch(Policy{}, kernel, acc, ridxs.data());
        }
      },
      matrix);
}

__device__ std::int32_t Laneid() {
  unsigned int laneid;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

template <typename Policy, typename RidxIterSpan>
__global__ __launch_bounds__(kBlockThreads) void ProducerConsumerKernel(
    EllpackDeviceAccessor const matrix, FeatureGroupsAccessor const feature_groups,
    RidxIterSpan* d_ridx_iters, common::Span<GradientPairInt64>* node_hists, bst_node_t n_nodes,
    linalg::MatrixView<GradientPair const> d_gpair,
    common::Span<GradientQuantiser const> roundings) {
  constexpr std::int32_t kWarpThreads = 32;
  static_assert(Policy::kBlockThreads % kWarpThreads == 0);
  constexpr std::int32_t kWarps = Policy::kBlockThreads / kWarpThreads;
  static_assert(Policy::kBlockThreads % 2 == 0);
  // half of the warps(threads) are used as consumer.
  constexpr std::int32_t kTileSize = Policy::kBlockThreads / 2;
  // 2 buffers for each warp, each buffer requires 2 barrier
  // We use half of the warps as producer
  // in total, we have kWarps * 2 barriers.
  static_assert(kWarps % 2 == 0);
  constexpr std::int32_t kProducers = kWarps / 2;
  constexpr std::int32_t kBuffers = kProducers * 2;
  constexpr std::int32_t kBarriers = kBuffers * 2;

  auto const lane_id = Laneid();
  auto const warp_id = static_cast<std::int32_t>(threadIdx.x) / kWarpThreads;
  bool const is_consumer = !(warp_id & 1);  // warp_id %2 == 0

  using Barrier = cuda::barrier<cuda::thread_scope_block>;
  __shared__ Barrier barriers[kBarriers];

  // Consumer signals data has been consumed
  Barrier* consumed = barriers;
  // Producer signals data has been produced
  Barrier* filled = consumed + kBuffers;

  if (threadIdx.x < kBarriers) {
    init(barriers + threadIdx.x, kWarpThreads * 2);
  }

  auto nidx_in_set = blockIdx.z;
  auto const n_targets = d_gpair.Shape(1);
  auto const d_ridx = d_ridx_iters[nidx_in_set];
  FeatureGroup const group = feature_groups[blockIdx.y];
  std::int32_t const feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;
  std::int32_t const n_elements = feature_stride * d_ridx.size();

  // grid stride loop
  std::int32_t const kStride = gridDim.x * kTileSize;
  // first grid

  extern __shared__ __align__(16) char shmem[];
  bst_bin_t* bufs[2]{reinterpret_cast<bst_bin_t*>(shmem),
                     reinterpret_cast<bst_bin_t*>(shmem) + kTileSize};

  auto d_roundings = roundings.data();
  // fixme: align
  GradientPairInt64* node_hist = reinterpret_cast<GradientPairInt64*>(bufs[1] + kTileSize);
  // Initialize the shared memory for the partial histogram
  dh::BlockFill(node_hist, group.num_bins, GradientPairInt64{});

  __syncthreads();

  auto calc_valid_items = [&](std::int32_t offset) {
    return cuda::std::min(n_elements - offset, kTileSize);
  };

  // Part ii of the consumer
  auto calc_idx = [&](auto full_tile, std::int32_t valid_items, std::int32_t offset,
                      bst_bin_t* buf) {
    std::int32_t tidx = warp_id / 2 * kWarpThreads + lane_id;
    std::int32_t const idx = offset + tidx;
    if (full_tile || tidx < valid_items) {
      cuda_impl::RowIndexT ridx = d_ridx[idx / feature_stride];
      auto fidx = FeatIdx(group, idx, feature_stride);
      std::int32_t iidx = IterIdx(matrix, ridx, fidx);  // fixme, u64 int
      buf[tidx] = iidx;
    }
  };
  // Part i of the consumer
  auto process_gidx = [&](auto full_tile, auto valid_items, std::int32_t offset,
                          bst_bin_t const* buf) {
    std::int32_t tidx = warp_id / 2 * kWarpThreads + lane_id;
    std::int32_t const idx = offset + tidx;

    if (full_tile || tidx < valid_items) {
      auto compressed_bin = buf[tidx];
      if (compressed_bin != matrix.NullValue()) {
        if (Policy::kCompressed) {
          auto fidx = FeatIdx(group, idx, feature_stride);
          compressed_bin += matrix.feature_segments[fidx];
        }
        cuda_impl::RowIndexT ridx = d_ridx[idx / feature_stride];
        compressed_bin *= n_targets;  // fixme (group.start_bin)
        // TODO(jiamingy): Assign a thread for each target.
        for (bst_target_t t = 0; t < n_targets; ++t) {
          auto adjusted = d_roundings[t].ToFixedPoint(d_gpair(ridx, t));
          AtomicAddGpairShared(node_hist + compressed_bin - group.start_bin, adjusted);
        }
      }
    }
  };

  // The producer
  auto load_gidx = [&](auto full_tile, auto valid_items, std::int32_t offset, bst_bin_t* buf) {
    // the thread index of the corresponding consumer
    std::int32_t tidx = (warp_id - 1) / 2 * kWarpThreads + lane_id;
    std::int32_t const idx = offset + tidx;
    if (full_tile || idx < valid_items) {
      buf[tidx] = matrix.gidx_iter[buf[tidx]];
    }
  };

  auto initial_consume = [&](auto offset, bst_bin_t* buf) {
    std::int32_t const valid_items = calc_valid_items(offset);
    if (kTileSize == valid_items) {
      calc_idx(std::true_type{}, valid_items, offset, buf);
    } else {
      calc_idx(std::false_type{}, valid_items, offset, buf);
    }
  };

  std::int32_t n_stages = common::DivRoundUp(n_elements, kStride);

  auto consumer = [&] {
    std::int32_t offset = blockIdx.x * kTileSize;
    // Calculate the index for the first buffer
    initial_consume(offset, bufs[0]);
    // Signal the first buffer is ready for the initial fill
    [[maybe_unused]] auto token0 = consumed[warp_id].arrive();
    __syncthreads();

    // Calculate the index for the second buffer
    if (offset + kStride < n_elements) {
      initial_consume(offset + kStride, bufs[1]);
    }
    // Signal the second buffer is ready for the initial fill
    [[maybe_unused]] auto token1 = consumed[warp_id + 1].arrive();

    std::int32_t stage = 0;

    for (std::int32_t j = 0; j < n_stages; ++j) {
      auto offset = j * kStride + blockIdx.x * kTileSize;

      std::int32_t const valid_items = calc_valid_items(offset);
      // wait for buffer to be ready to use.
      filled[warp_id + stage].arrive_and_wait();
      if (kTileSize == valid_items) {
        process_gidx(std::true_type{}, valid_items, offset, bufs[stage]);
      } else {
        process_gidx(std::false_type{}, valid_items, offset, bufs[stage]);
      }

      // Calculate the idx using the same buffer (stage)
      if (j != (n_stages - 1)) {
        auto offset = (j + 1) * kStride + blockIdx.x * kTileSize;
        if (kTileSize == valid_items) {
          calc_idx(std::true_type{}, valid_items, offset, bufs[stage]);
        } else {
          calc_idx(std::false_type{}, valid_items, offset, bufs[stage]);
        }
      }

      // signal buffer is used, ready for filling.
      [[maybe_unused]] auto token = consumed[warp_id + stage].arrive();

      stage ^= 1;
    }
  };

  auto producer = [&] {
    std::int32_t stage = 0;
    for (std::int32_t j = 0; j < n_stages; ++j) {
      auto offset = j * kStride + blockIdx.x * kTileSize;
      std::int32_t const valid_items = calc_valid_items(offset);

      // wait for the consumer to consume the data
      consumed[stage + warp_id - 1].arrive_and_wait();
      if (j == 0) {
        __syncthreads();
      }
      if (kTileSize == valid_items) {
        load_gidx(std::true_type{}, valid_items, offset, bufs[stage]);
      } else {
        load_gidx(std::false_type{}, valid_items, offset, bufs[stage]);
      }
      // signal the data is ready for consumption
      [[maybe_unused]] auto token = filled[stage + warp_id - 1].arrive();

      stage ^= 1;
    }
  };

  if (is_consumer) {
    consumer();
  } else {
    producer();
  }

  // Write shared memory back to global memory
  __syncthreads();

  auto d_node_hist = node_hists[nidx_in_set].data();

  for (auto i : dh::BlockStrideRange(0, group.num_bins)) {
    // fixme: n targets, need to handle it in the feature groups as well.
    if (node_hist[i].GetQuantisedHess() == -1) {
      AtomicAddGpairGlobal(d_node_hist + group.start_bin + i, node_hist[i]);
    }
  }
}

void DeviceHistogramBuilder::BuildHistogramPC(CUDAContext const* ctx, EllpackAccessor const& matrix,
                                              FeatureGroupsAccessor const& feature_groups,
                                              linalg::MatrixView<GradientPair const> gpair,
                                              common::Span<common::Span<const std::uint32_t>> ridxs,
                                              common::Span<common::Span<GradientPairInt64>> hists,
                                              std::size_t n_max_samples,
                                              common::Span<GradientQuantiser const> roundings) {
  using Policy = HistPolicy<kBlockThreads, 1, true>;
  constexpr std::int32_t kTileSize = kBlockThreads / 2;

  auto n_nodes = hists.size();

  auto acc = std::get<EllpackDeviceAccessor>(matrix);
  std::cout << "n_groups:" << feature_groups.NumGroups() << std::endl;
  int columns_per_group = common::DivRoundUp(acc.row_stride, feature_groups.NumGroups());
  std::size_t items_per_group = n_max_samples * columns_per_group;

  auto n_grids = common::DivRoundUp(items_per_group, kTileSize);
  dim3 conf(n_grids, feature_groups.NumGroups(), n_nodes);
  std::size_t shmem_bytes = feature_groups.ShmemSize() + sizeof(bst_bin_t) * Policy::kBlockThreads;

  using RidxIter = thrust::counting_iterator<cuda_impl::RowIndexT>;
  dh::caching_device_vector<common::IterSpan<RidxIter>> ridx_iters(
      hists.size(), common::IterSpan{thrust::make_counting_iterator(0u), gpair.Shape(0)});

  auto kernel = ProducerConsumerKernel<Policy, common::IterSpan<RidxIter>>;
  kernel<<<conf, Policy::kBlockThreads, shmem_bytes, ctx->Stream()>>>(
      acc, feature_groups, ridx_iters.data().get(), hists.data(), hists.size(), gpair, roundings);
  dh::safe_cuda(cudaPeekAtLastError());
}

void DeviceHistogramBuilder::AllReduceHist(Context const* ctx, MetaInfo const& info,
                                           bst_node_t nidx, std::size_t num_histograms) {
  this->monitor_.Start(__func__);
  auto d_node_hist = hist_.GetNodeHistogram(nidx);
  using ReduceT = typename std::remove_pointer<decltype(d_node_hist.data())>::type::ValueT;
  auto rc = collective::GlobalSum(
      ctx, info,
      linalg::MakeVec(reinterpret_cast<ReduceT*>(d_node_hist.data()),
                      d_node_hist.size() * 2 * num_histograms, ctx->Device()));
  SafeColl(rc);
  this->monitor_.Stop(__func__);
}
}  // namespace xgboost::tree
