/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <ctgmath>
#include <limits>

#include "xgboost/base.h"
#include "row_partitioner.cuh"

#include "histogram.cuh"

#include "../../data/ellpack_page.cuh"
#include "../../common/device_helpers.cuh"
#include "histogram.cuh"

namespace xgboost {
namespace tree {
// Following 2 functions are slightly modifed version of fbcuda.

/* \brief Constructs a rounding factor used to truncate elements in a sum such that the
   sum of the truncated elements is the same no matter what the order of the sum is.

 * Algorithm 5: Reproducible Sequential Sum in 'Fast Reproducible Floating-Point
 * Summation' by Demmel and Nguyen

 * In algorithm 5 the bound is calculated as $max(|v_i|) * n$.  Here we use the bound
 *
 * \begin{equation}
 *   max( fl(\sum^{V}_{v_i>0}{v_i}), fl(\sum^{V}_{v_i<0}|v_i|) )
 * \end{equation}
 *
 * to avoid outliers, as the full reduction is reproducible on GPU with reduction tree.
 */
template <typename T>
XGBOOST_DEV_INLINE __host__ T CreateRoundingFactor(T max_abs, int n) {
  T delta = max_abs / (static_cast<T>(1.0) - 2 * n * std::numeric_limits<T>::epsilon());

  // Calculate ceil(log_2(delta)).
  // frexpf() calculates exp and returns `x` such that
  // delta = x * 2^exp, where `x` in (-1.0, -0.5] U [0.5, 1).
  // Because |x| < 1, exp is exactly ceil(log_2(delta)).
  int exp;
  std::frexp(delta, &exp);

  // return M = 2 ^ ceil(log_2(delta))
  return std::ldexp(static_cast<T>(1.0), exp);
}

namespace {
struct Pair {
  GradientPair first;
  GradientPair second;
};
XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}
}  // anonymous namespace

struct Clip : public thrust::unary_function<GradientPair, Pair> {
  static XGBOOST_DEV_INLINE float Pclip(float v) {
    return v > 0 ? v : 0;
  }
  static XGBOOST_DEV_INLINE float Nclip(float v) {
    return v < 0 ? abs(v) : 0;
  }

  XGBOOST_DEV_INLINE Pair operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return { GradientPair{ pg, ph }, GradientPair{ ng, nh } };
  }
};

template <typename GradientSumT>
GradientSumT CreateRoundingFactor(common::Span<GradientPair const> gpair) {
  using T = typename GradientSumT::ValueT;
  dh::XGBCachingDeviceAllocator<char> alloc;

  thrust::device_ptr<GradientPair const> gpair_beg {gpair.data()};
  thrust::device_ptr<GradientPair const> gpair_end {gpair.data() + gpair.size()};
  auto beg = thrust::make_transform_iterator(gpair_beg, Clip());
  auto end = thrust::make_transform_iterator(gpair_end, Clip());
  Pair p = thrust::reduce(thrust::cuda::par(alloc), beg, end, Pair{});
  GradientPair positive_sum {p.first}, negative_sum {p.second};

  auto histogram_rounding = GradientSumT {
    CreateRoundingFactor<T>(std::max(positive_sum.GetGrad(), negative_sum.GetGrad()),
                            gpair.size()),
    CreateRoundingFactor<T>(std::max(positive_sum.GetHess(), negative_sum.GetHess()),
                            gpair.size()) };
  return histogram_rounding;
}

template GradientPairPrecise CreateRoundingFactor(common::Span<GradientPair const> gpair);
template GradientPair CreateRoundingFactor(common::Span<GradientPair const> gpair);

template <typename GradientSumT>
__global__ void SharedMemHistKernel(EllpackDeviceAccessor matrix,
                                    FeatureGroupsAccessor feature_groups,
                                    common::Span<const RowPartitioner::RowIndexT> d_ridx,
                                    GradientSumT* __restrict__ d_node_hist,
                                    const GradientPair* __restrict__ d_gpair,
                                    GradientSumT const rounding,
                                    bool use_shared_memory_histograms) {
  using T = typename GradientSumT::ValueT;
  extern __shared__ char smem[];
  FeatureGroup group = feature_groups[blockIdx.y];
  GradientSumT* smem_arr = reinterpret_cast<GradientSumT*>(smem);  // NOLINT
  if (use_shared_memory_histograms) {
    dh::BlockFill(smem_arr, group.num_bins, GradientSumT());
    __syncthreads();
  }
  int feature_stride = matrix.is_dense ? group.num_features : matrix.row_stride;
  size_t n_elements = feature_stride * d_ridx.size();
  for (auto idx : dh::GridStrideRange(static_cast<size_t>(0), n_elements)) {
    int ridx = d_ridx[idx / feature_stride];
    int gidx = matrix.gidx_iter[ridx * matrix.row_stride + group.start_feature +
                                idx % feature_stride];
    if (gidx != matrix.NumBins()) {
      GradientSumT truncated {
        TruncateWithRoundingFactor<T>(rounding.GetGrad(), d_gpair[ridx].GetGrad()),
        TruncateWithRoundingFactor<T>(rounding.GetHess(), d_gpair[ridx].GetHess()),
      };
      // If we are not using shared memory, accumulate the values directly into
      // global memory
      GradientSumT* atomic_add_ptr =
        use_shared_memory_histograms ? smem_arr : d_node_hist;
      gidx = use_shared_memory_histograms ? gidx - group.start_bin : gidx;
      dh::AtomicAddGpair(atomic_add_ptr + gidx, truncated);
    }
  }

  if (use_shared_memory_histograms) {
    // Write shared memory back to global memory
    __syncthreads();
    for (auto i : dh::BlockStrideRange(0, group.num_bins)) {
      GradientSumT truncated{
          TruncateWithRoundingFactor<T>(rounding.GetGrad(),
                                        smem_arr[i].GetGrad()),
          TruncateWithRoundingFactor<T>(rounding.GetHess(),
                                        smem_arr[i].GetHess()),
      };
      dh::AtomicAddGpair(d_node_hist + group.start_bin + i, truncated);
    }
  }
}

template <typename GradientSumT>
GPUHistogramBuilder<GradientSumT>::GPUHistogramBuilder(
    FeatureGroupsAccessor const &feature_groups, uint32_t tuning_threads) {
  int device = cub::CurrentDevice();
  size_t smem_size = sizeof(GradientSumT) * feature_groups.max_group_bins;
  auto kernel = SharedMemHistKernel<GradientSumT>;

  int32_t sm_version;
  cub::SmVersion(sm_version);
  sm_version /= 10;

  auto default_block_threads = [smem_size]() {
    int min_grid_size;
    int block_threads = -1;
    auto kernel = SharedMemHistKernel<GradientSumT>;
    dh::safe_cuda(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_threads, kernel, smem_size, 0));
    CHECK_NE(block_threads, -1);
    return static_cast<uint32_t>(block_threads);
  };

  switch (sm_version) {
  case 35: {
    block_threads_ = default_block_threads();
    break;
  }
  case 50: {
    block_threads_ = default_block_threads();
    break;
  }
  case 52: {
    block_threads_ = default_block_threads();
    break;
  }
  case 60: {
    block_threads_ = default_block_threads();
    break;
  }
  case 61:
    block_threads_ = 256;
    break;
  case 70: {
    block_threads_ = default_block_threads();
    break;
  }
  case 75: {
    block_threads_ = default_block_threads();
    break;
  }
  case 80: {
    block_threads_ = default_block_threads();
    break;
  }
  default: {
    block_threads_ = default_block_threads();
    break;
  }
  }

  if (tuning_threads != 0) {
    block_threads_ = tuning_threads;
  }

  int num_groups = feature_groups.NumGroups();
  int n_mps = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, device));
  int n_blocks_per_mp = 0;
  dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor
                (&n_blocks_per_mp, kernel, block_threads_, smem_size));
  unsigned grid_size = n_blocks_per_mp * n_mps;

  int num_groups_threshold = 4;
  grid_size = common::DivRoundUp(grid_size,
      common::DivRoundUp(num_groups, num_groups_threshold));

  grids_ = dim3(grid_size, num_groups);
  smem_size_ = smem_size;

  shared_ = smem_size_ <= dh::MaxSharedMemory(device);;
  smem_size_ = shared_ ? smem_size_ : 0;
}

template <typename GradientSumT>
void GPUHistogramBuilder<GradientSumT>::Build(
    EllpackDeviceAccessor const &matrix,
    FeatureGroupsAccessor const &feature_groups,
    common::Span<GradientPair const> gpair, common::Span<const uint32_t> ridx,
    common::Span<GradientSumT> histogram, GradientSumT rounding) {
  dh::LaunchKernel{grids_, block_threads_, smem_size_}(
      SharedMemHistKernel<GradientSumT>, matrix, feature_groups, ridx,
      histogram.data(), gpair.data(), rounding, shared_);
}

template class GPUHistogramBuilder<GradientPairPrecise>;
template class GPUHistogramBuilder<GradientPair>;
}  // namespace tree
}  // namespace xgboost
