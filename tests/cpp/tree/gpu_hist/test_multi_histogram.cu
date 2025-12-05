/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>

#include <cuda/functional>

#include "../../../../src/common/device_debug.cuh"
#include "../../../../src/tree/gpu_hist/histogram.cuh"
#include "../../helpers.h"
#include "../../histogram_helpers.h"
#include "dummy_quantizer.cuh"  // for MakeDummyQuantizers

namespace xgboost::tree::cuda_impl {
TEST(GpuMultiHistogram, Basic) {
  auto ctx = MakeCUDACtx(0);
  bst_bin_t n_bins = 16;
  bst_target_t n_targets = 2;
  bst_feature_t n_features = 4;

  bst_idx_t n_samples = 64;
  auto page = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins);

  auto cuts = page->CutsShared();

  FeatureGroups fg{*cuts, true, std::numeric_limits<std::size_t>::max()};
  auto fg_acc = fg.DeviceAccessor(ctx.Device());

  DeviceHistogramBuilder histogram;
  bst_bin_t n_total_bins = n_targets * n_features * n_bins;
  histogram.Reset(&ctx, /*max_cached_hist_nodes=*/2, fg_acc, n_total_bins, false);

  auto gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);
  dh::device_vector<std::uint32_t> ridx(n_samples);
  thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);

  histogram.AllocateHistograms(&ctx, {0});
  auto node_hist = histogram.GetNodeHistogram(0);
  auto quantizers = MakeDummyQuantizers(n_targets);

  histogram.BuildHistogram(ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), fg_acc,
                           gpairs.View(ctx.Device()), dh::ToSpan(ridx), node_hist,
                           dh::ToSpan(quantizers));

  // std::vector<GradientPairInt64> h_node_hist(node_hist.size());
  // dh::CopyDeviceSpanToVector(&h_node_hist, node_hist);
  // // The values are evenly distributed across all bins
  // auto expected = n_samples / n_bins;
  // for (auto v : h_node_hist) {
  //   ASSERT_EQ(v.GetQuantisedGrad(), expected);
  //   ASSERT_EQ(v.GetQuantisedHess(), expected);
  // }
}

namespace {
XGBOOST_DEV_INLINE bst_feature_t FeatIdx(bst_idx_t idx, std::int32_t feature_stride) {
  auto fidx = idx % feature_stride;
  return fidx;
}

template <typename IterT>
XGBOOST_DEV_INLINE bst_idx_t IterIdx(EllpackAccessorImpl<IterT> const& matrix, std::uint32_t ridx,
                                     bst_feature_t fidx) {
  // We can pre-calculate the multiplication if necessary.
  return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
}

// rtx4070tis, 537MB, 122.07GB/s
__global__ void TestHistBuildKernel(EllpackDeviceAccessor matrix,
                                    common::Span<GradientPairInt64> d_node_hist,
                                    common::Span<std::uint32_t> d_ridx,
                                    common::Span<GradientQuantiser const> roundings) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_elements) {
    return;
  }
  std::uint32_t ridx = tid / matrix.row_stride;
  auto fidx = FeatIdx(tid, matrix.row_stride);
  auto idx = IterIdx(matrix, ridx, fidx);
  bst_bin_t compressed_bin = matrix.gidx_iter[idx];
  if (compressed_bin == -1) {
    printf("-1\n");
  }
}

// rtx4070tis 186.93GB/s
__global__ void RawReadKernel(EllpackDeviceAccessor matrix, common::Span<std::uint32_t> d_ridx) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_elements) {
    return;
  }
  bst_bin_t compressed_bin = matrix.gidx_iter[tid];
  if (compressed_bin == -1) {
    printf("-1\n");
  }
}

// rtx4070tis 4-IPT, 152GB/s, same ballpart with 8-IPT
//
// Without the gidx_iter read, this kernel has about 242GB/s throughput, most of the
// overhead comes from the integer multiplication inside IterIdx.
template <std::int32_t kItemsPerThread, std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void ReadUnrollKernel(
    EllpackDeviceAccessor matrix, common::Span<GradientPairInt64> d_node_hist,
    common::Span<std::uint32_t const> d_ridx, common::Span<GradientQuantiser const> roundings) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  constexpr auto kItemsPerTile = kItemsPerThread * kBlockThreads;

  std::size_t idx[kItemsPerThread];
  std::uint32_t ridx[kItemsPerThread];
  bst_bin_t gidx[kItemsPerThread];

  auto load = [&](std::size_t offset) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      idx[i] = offset + i * kBlockThreads + threadIdx.x;
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      ridx[i] = idx[i] / matrix.row_stride;
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      auto fidx = FeatIdx(idx[i], matrix.row_stride);
      gidx[i] = matrix.gidx_iter[IterIdx(matrix, ridx[i], fidx)];
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      auto compressed_bin = gidx[i];
      if (compressed_bin == -1) {
        printf("-1\n");
      }
    }
  };

  std::size_t offset = blockIdx.x * kItemsPerTile;
  while (offset + kItemsPerTile < n_elements) {
    load(offset);
    offset += kItemsPerTile * gridDim.x;
  }
}

XGBOOST_DEV_INLINE void AtomicAddGpairShared(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<int64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  AtomicAdd64As32(dst_ptr, g);
  AtomicAdd64As32(dst_ptr + 1, h);
}

template <std::int32_t kItemsPerThread, std::int32_t kBlockThreads>
__global__ void RawReadUnrollKernel(EllpackDeviceAccessor matrix,
                                    common::Span<std::uint32_t const> d_ridx) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  constexpr auto kItemsPerTile = kItemsPerThread * kBlockThreads;

  bst_bin_t gidx[kItemsPerThread];

  auto load = [&](std::size_t offset) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      auto idx = offset + i * kBlockThreads + threadIdx.x;
      gidx[i] = matrix.gidx_iter[idx];
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      auto compressed_bin = gidx[i];
      if (compressed_bin == -1) {
        printf("-1\n");
      }
    }
  };

  std::size_t offset = blockIdx.x * kItemsPerTile;
  while (offset + kItemsPerTile < n_elements) {
    load(offset);
    offset += kItemsPerTile * gridDim.x;
  }
}

// ncu --set full --kernel-name PrefetchReadKernel --launch-skip 0 --launch-count 1 -o pf-tis-%i
// ./testxgboost --gtest_filter="MicroBenchHist.PrefetchRead"
template <std::int32_t kItemsPerThread, std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void PrefetchReadKernel(
    EllpackDeviceAccessor matrix, common::Span<std::uint32_t const> d_ridx) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();

  constexpr std::int32_t kPrefetch = 4;

  constexpr auto kItemsPerTile = kBlockThreads;
  std::size_t offset = blockIdx.x * kItemsPerTile;
  auto stride = kItemsPerTile * gridDim.x;

  // bst_idx_t indices[kItemsPerThread];

  auto prefetch = [&](std::size_t offset, std::int32_t k) {
    auto idx = offset + (k * stride) + (kItemsPerThread * stride) + threadIdx.x;
    if (idx < n_elements) {
      matrix.gidx_iter.Prefetch(idx);
    }
  };
  auto load = [&](std::size_t offset, std::int32_t k) {
    auto idx = offset + (k * stride) + threadIdx.x;
    if (idx < n_elements) {
      return matrix.gidx_iter[idx];
    }
    return 0u;
  };

  while (offset + kItemsPerTile < n_elements) {
#pragma unroll kItemsPerThread
    for (std::int32_t k = 0; k < kItemsPerThread; ++k) {
      prefetch(offset, k);
      auto compressed_bin = load(offset, k);
      if (compressed_bin == -1) {
        printf("-1\n");
      }
    }

    offset += stride * kItemsPerThread;
  }
}

template <std::int32_t kItemsPerThread, std::int32_t kBlockThreads>
__global__ __launch_bounds__(kBlockThreads) void PrefetchReadTileKernel(
    EllpackDeviceAccessor matrix, common::Span<std::uint32_t const> d_ridx) {
  auto n_elements = matrix.row_stride * d_ridx.size();

  std::int32_t constexpr kTileSize = kItemsPerThread * kBlockThreads;
  std::size_t const offset = blockIdx.x * kTileSize;
  std::int32_t const valid_items =
      cuda::std::min(n_elements - offset, static_cast<std::size_t>(kTileSize));

  size_t start_bytes[kItemsPerThread];
  auto prefetch_tile = [&](auto full_tile) {
    for (int j = 0; j < kItemsPerThread; ++j) {
      const int idx = j * kBlockThreads + threadIdx.x;
      if (full_tile || idx < valid_items) {
        start_bytes[j] = matrix.gidx_iter.Prefetch(offset + idx);
      }
    }
  };
  auto process_tile = [&](auto full_tile) {
    bst_bin_t gidx[kItemsPerThread];

    for (int j = 0; j < kItemsPerThread; ++j) {
      // block strided loop
      const int idx = j * kBlockThreads + threadIdx.x;
      if (full_tile || idx < valid_items) {
        gidx[j] = matrix.gidx_iter.Read(start_bytes[j]);
      }
    }

    for (int j = 0; j < kItemsPerThread; ++j) {
      if (gidx[j] == -1) {
        printf("-1\n");
      }
    }
  };

  if (kTileSize == valid_items) {
    prefetch_tile(::cuda::std::true_type{});
    process_tile(::cuda::std::true_type{});
  } else {
    prefetch_tile(::cuda::std::false_type{});
    process_tile(::cuda::std::false_type{});
  }
}

// rtx4070tis 37.89GB/s 100 occupancy
template <std::int32_t kItemsPerThread, std::int32_t kBlockThreads>
__global__ void ReadSharedAddUnrollKernel(EllpackDeviceAccessor matrix,
                                          common::Span<std::uint32_t const> d_ridx) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  constexpr auto kItemsPerTile = kItemsPerThread * kBlockThreads;

  bst_bin_t gidx[kItemsPerThread];

  extern __align__(16) __shared__ char shmem[];
  auto node_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  auto load = [&](std::size_t offset) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      auto idx = offset + i * kBlockThreads + threadIdx.x;
      gidx[i] = matrix.gidx_iter[idx];
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      auto compressed_bin = gidx[i];
      AtomicAddGpairShared(node_hist + compressed_bin, GradientPairInt64{gidx[i], i});
    }
  };

  std::size_t offset = blockIdx.x * kItemsPerTile;
  while (offset + kItemsPerTile < n_elements) {
    load(offset);
    offset += kItemsPerTile * gridDim.x;
  }
}

// rtx4070tis 537MB, 30.18GB/s
__global__ void TestHistBuildKernelRowWise(EllpackDeviceAccessor matrix,
                                           common::Span<GradientPairInt64> d_node_hist,
                                           common::Span<std::uint32_t> d_ridx,
                                           common::Span<GradientQuantiser const> roundings) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= d_ridx.size()) {
    return;
  }
  // std::uint32_t ridx = tid / matrix.row_stride;
  for (auto fidx = 0; fidx < matrix.row_stride; ++fidx) {
    auto idx = IterIdx(matrix, tid, fidx);
    bst_bin_t compressed_bin = matrix.gidx_iter[idx];
    if (compressed_bin == -1) {
      printf("-1\n");
    }
  }
}

XGBOOST_DEV_INLINE void AtomicAddGpairGlobal(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<uint64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  atomicAdd(dst_ptr, *reinterpret_cast<uint64_t*>(&g));
  atomicAdd(dst_ptr + 1, *reinterpret_cast<uint64_t*>(&h));
}

// rtx4070tis Ellpack size / kernel duration: 35.58GB/s
__global__ void TestGlobalAtomicKernel(EllpackDeviceAccessor matrix,
                                       common::Span<GradientPairInt64> d_node_hist,
                                       common::Span<std::uint32_t> d_ridx,
                                       common::Span<GradientQuantiser const> roundings) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_elements) {
    return;
  }
  std::uint32_t ridx = tid / matrix.row_stride;
  auto fidx = FeatIdx(tid, matrix.row_stride);
  auto idx = IterIdx(matrix, ridx, fidx);
  auto g = GradientPairInt64{ridx, fidx};  // simulate
  AtomicAddGpairGlobal(d_node_hist.data() + idx % d_node_hist.size(), g);
}

// rtx4070tis 181GB/s
__global__ void TestSharedAtomicKernel(EllpackDeviceAccessor matrix,
                                       common::Span<GradientPairInt64> d_node_hist,
                                       common::Span<std::uint32_t> d_ridx,
                                       common::Span<GradientQuantiser const> roundings) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_elements) {
    return;
  }

  extern __align__(16) __shared__ char shmem[];
  auto node_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  std::uint32_t ridx = tid / matrix.row_stride;
  auto fidx = FeatIdx(tid, matrix.row_stride);
  auto g = GradientPairInt64{ridx, fidx};  // simulate
  AtomicAddGpairShared(node_hist + tid % 256, g);
}

class MicroBenchHist : public ::testing::Test {
 public:
  Context ctx{MakeCUDACtx(0)};

  bst_bin_t n_bins = 256;
  bst_target_t n_targets = 1;
  bst_feature_t n_features = 256;

  bool use_single_target = false;

  bst_idx_t n_samples = 1 << 21;

  std::unique_ptr<EllpackPageImpl> page;

  std::shared_ptr<common::HistogramCuts const> cuts;
  std::unique_ptr<FeatureGroups> p_fg;

  DeviceHistogramBuilder histogram;
  common::Span<GradientPairInt64> node_hist;
  linalg::Matrix<GradientPair> gpairs;
  dh::device_vector<std::uint32_t> ridx;
  dh::device_vector<GradientQuantiser> quantizers;

  static constexpr std::uint32_t kBlockThreads = 512;

  void SetUp() override {
    this->page = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins);
    this->cuts = page->CutsShared();

    this->p_fg = std::make_unique<FeatureGroups>(*cuts, true, dh::MaxSharedMemory(0));

    bst_bin_t n_total_bins = n_targets * n_features * n_bins;
    auto fg_acc = p_fg->DeviceAccessor(ctx.Device());
    histogram.Reset(&ctx, /*max_cached_hist_nodes=*/2, fg_acc, n_total_bins, false);

    gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);

    ridx.resize(n_samples);
    thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);

    histogram.AllocateHistograms(&ctx, {0});
    node_hist = histogram.GetNodeHistogram(0);

    quantizers = MakeDummyQuantizers(n_targets);
  }

  void BenchSharedAtomic() {
    auto n = page->Size() * page->info.row_stride;
    auto n_grids = common::DivRoundUp(n, kBlockThreads);
    auto n_bytes = sizeof(GradientPairInt64) * 256;

    TestSharedAtomicKernel<<<n_grids, kBlockThreads, n_bytes>>>(
        std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), node_hist,
        dh::ToSpan(ridx), dh::ToSpan(quantizers));
  }
  void BenchReadUnroll() {
    constexpr std::int32_t kItemsPerThread = 8;
    auto n = page->Size() * page->info.row_stride;
    auto n_grids = common::DivRoundUp(n, kBlockThreads) / 32;
    auto kernel = ReadUnrollKernel<kItemsPerThread, kBlockThreads>;
    std::cout << "n_grids:" << n_grids << std::endl;
    kernel<<<n_grids, kBlockThreads>>>(
        std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), node_hist,
        dh::ToSpan(ridx), dh::ToSpan(quantizers));
  }
  // tis: 385.23GB/s
  // dgx: 145.85GB/s
  // H200: 536.91GB/s
  void BenchRawReadUnroll() {
    constexpr std::int32_t kItemsPerThread = 8;
    auto n = page->Size() * page->info.row_stride;
    auto n_grids = common::DivRoundUp(n, kBlockThreads) / 64;
    auto kernel = RawReadUnrollKernel<kItemsPerThread, kBlockThreads>;
    std::cout << "n_grids:" << n_grids << std::endl;
    auto n_bytes = sizeof(GradientPairInt64) * 256;
    kernel<<<n_grids, kBlockThreads, n_bytes>>>(
        std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), dh::ToSpan(ridx));
  }
  // tis: 242.45GB/s
  // H200: 233.05GB/s!!!!
  // DGX: 117.59GB/s
  void BenchPrefetchRead() {
    constexpr std::int32_t kItemsPerThread = 4;
    auto n = page->Size() * page->info.row_stride;
    auto n_grids = common::DivRoundUp(n, kBlockThreads) / 64;
    std::cout << "n_grids:" << n_grids << " n:" << n << std::endl;
    auto kernel = PrefetchReadKernel<kItemsPerThread, kBlockThreads>;
    kernel<<<n_grids, kBlockThreads>>>(
        std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), dh::ToSpan(ridx));
  }
  // tis 388GB/s no prefetch
  // tis 406.28GB/s prefetch
  // h200 651.99GB/s no prefetch
  // h200 455.48GB/s prefetch !!!
  // dgx 171.57GB/s no prefetch
  // dgx 180GB/s prefetch
  void BenchPrefetchTile() {
    constexpr std::int32_t kItemsPerThread = 4;
    auto n = page->Size() * page->info.row_stride;
    auto n_grids = common::DivRoundUp(n, kBlockThreads * kItemsPerThread);
    auto kernel = PrefetchReadTileKernel<kItemsPerThread, kBlockThreads>;
    kernel<<<n_grids, kBlockThreads>>>(
        std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), dh::ToSpan(ridx));
  }
  // H200: 1.02T/s
  // DGX: 108GB/s
  // tis: 314.58GB/s
  // H200 (copyable): 1.55T/s
  // DGX (copyable): 115.87GB/s
  // tis (copyable): 314.48GB/s
  void BenchThrustTransform() {
    auto const& iter = page->gidx_buffer;
    dh::device_vector<char> tmp(iter.size_bytes());
    thrust::transform(iter.data(), iter.data() + iter.size_bytes(), tmp.data(),
                      cuda::proclaim_copyable_arguments(
                          [] XGBOOST_DEVICE(common::CompressedByteT b) { return b + 1; }));
  }
  // H200: 580.90GB/s
  // DGX: 103GB/s
  // tis: 259.29GB/s
  // H200 (copyable): 581.96GB/s
  // tis (copyable): 259.07GB/s
  void BenchThrustTransformCntIter() {
    auto const& iter = page->gidx_buffer.data();
    dh::device_vector<char> tmp(page->gidx_buffer.size_bytes());
    auto it = dh::MakeIndexTransformIter([=] XGBOOST_DEVICE(std::size_t i) { return iter[i]; });
    thrust::transform(it, it + page->gidx_buffer.size_bytes(), tmp.data(),
                      cuda::proclaim_copyable_arguments(
                          [] XGBOOST_DEVICE(common::CompressedByteT b) { return b + 1; }));
  }
  // tis 240.37GB/s
  // dgx 107GB/s
  // h200 402.18GB/s
  void BenchForEachIter() {
    auto acc = std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {}));

    auto const& iter = page->gidx_buffer.data();
    auto it = thrust::make_counting_iterator(0ul);
    thrust::for_each_n(it, page->n_rows * page->info.row_stride, [=] XGBOOST_DEVICE(std::size_t i) {
      bst_bin_t gidx = acc.gidx_iter[i];
      if (gidx == -1) {
        printf("-1\n");
      }
    });
  }

  void BenchBuild() {
    auto ridxs = dh::device_vector<common::Span<std::uint32_t const>>{dh::ToSpan(ridx)};
    auto hists = dh::device_vector<common::Span<GradientPairInt64>>{node_hist};
    std::cout << "gpair size:" << common::HumanMemUnit(this->gpairs.Size() * sizeof(GradientPair))
              << std::endl;
    this->histogram.BuildHistogram(this->ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}),
                                   p_fg->DeviceAccessor(ctx.Device()),
                                   this->gpairs.View(this->ctx.Device()), dh::ToSpan(ridxs),
                                   dh::ToSpan(hists), ridx.size(), dh::ToSpan(this->quantizers));
  }

  void BenchStBuild() {
    GradientQuantiser q{GradientPairPrecise{1.0f, 1.0f}, GradientPairPrecise{1.0f, 1.0f}};
    this->histogram.BuildHistogram(
        this->ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), p_fg->DeviceAccessor(ctx.Device()),
        this->gpairs.Data()->ConstDeviceSpan(), dh::ToSpan(ridx), this->node_hist, q);
  }
};
}  // namespace

TEST(GpuMultiHistogram, Large) {
  auto ctx = MakeCUDACtx(0);

  bst_bin_t n_bins = 256;
  bst_target_t n_targets = 1;
  bst_feature_t n_features = 256;

  bool use_single_target = false;

  bst_idx_t n_samples = 1 << 21;
  auto page = MakeEllpackForTest(&ctx, n_samples, n_features, n_bins);

  auto cuts = page->CutsShared();

  FeatureGroups fg{
      *cuts, true,
      use_single_target ? dh::MaxSharedMemoryOptin(0) : std::numeric_limits<std::size_t>::max()};
  auto fg_acc = fg.DeviceAccessor(ctx.Device());

  DeviceHistogramBuilder histogram;
  bst_bin_t n_total_bins = n_targets * n_features * n_bins;
  histogram.Reset(&ctx, /*max_cached_hist_nodes=*/2, fg_acc, n_total_bins, !use_single_target);

  auto gpairs = linalg::Constant(&ctx, GradientPair{1.0f, 1.0f}, n_samples, n_targets);
  dh::device_vector<std::uint32_t> ridx(n_samples);
  thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);

  histogram.AllocateHistograms(&ctx, {0});
  auto node_hist = histogram.GetNodeHistogram(0);

  auto quantizers = MakeDummyQuantizers(n_targets);
  constexpr std::uint32_t kBlockThreads = 512;
  // {
  //   auto n = page->Size() * page->info.row_stride;
  //   auto n_grids = common::DivRoundUp(n, kBlockThreads);

  //   TestHistBuildKernel<<<n_grids, kBlockThreads>>>(
  //       std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), node_hist,
  //       dh::ToSpan(ridx), dh::ToSpan(quantizers));
  // }
  // {
  //   auto n = page->Size() * page->info.row_stride;
  //   auto n_grids = common::DivRoundUp(n, kBlockThreads);

  //   RawReadKernel<<<n_grids, kBlockThreads>>>(
  //       std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), dh::ToSpan(ridx));
  // }
  // {
  //   auto n = page->Size();
  //   auto n_grids = common::DivRoundUp(n, kBlockThreads);
  //   TestHistBuildKernelRowWise<<<n_grids, kBlockThreads>>>(
  //       std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), node_hist,
  //       dh::ToSpan(ridx), dh::ToSpan(quantizers));
  // }
  // {
  //   auto n = page->Size() * page->info.row_stride;
  //   auto n_grids = common::DivRoundUp(n, kBlockThreads);

  //   TestGlobalAtomicKernel<<<n_grids, kBlockThreads>>>(
  //       std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), node_hist,
  //       dh::ToSpan(ridx), dh::ToSpan(quantizers));
  // }
  // {
  //   constexpr std::int32_t kItemsPerThread = 8;
  //   auto n = page->Size() * page->info.row_stride;
  //   auto n_grids = common::DivRoundUp(n, kBlockThreads) / 64;
  //   auto kernel = ReadSharedAddUnrollKernel<kItemsPerThread, kBlockThreads>;
  //   auto n_bytes = sizeof(GradientPairInt64) * n_bins;
  //   kernel<<<n_grids, kBlockThreads, n_bytes>>>(
  //       std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), dh::ToSpan(ridx));
  // }
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, SharedAtomic) {
  this->BenchSharedAtomic();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, ReadUnroll) {
  this->BenchReadUnroll();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, RawReadUnroll) {
  this->BenchRawReadUnroll();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, PrefetchRead) {
  this->BenchPrefetchRead();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, PrefetchTile) {
  this->BenchPrefetchTile();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, ThrustTransform) {
  this->BenchThrustTransform();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, ThrustTransformCntIter) {
  this->BenchThrustTransformCntIter();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, ForEachIter) {
  this->BenchForEachIter();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, BuildPrefetch) {
  this->BenchBuild();
  debug::SyncDevice();
}

TEST_F(MicroBenchHist, StBuild) {
  this->BenchStBuild();
  debug::SyncDevice();
}
}  // namespace xgboost::tree::cuda_impl
