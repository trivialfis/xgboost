/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>

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
  histogram.Reset(&ctx, /*max_cached_hist_nodes=*/2, fg_acc, n_total_bins, true);

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
  // ridx_local = ridx - base_rowid  <== Row index local to each batch
  // entry_idx = ridx_local * row_stride <== Starting entry index for this row in the matrix
  // entry_idx += start_feature  <== Inside a row, first column inside this feature group
  // idx % feature_stride <== The feaature index local to the current feature group
  // entry_idx += idx % feature_stride <== Final index.
  return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
}

__global__ void TestHistBuildKernel(EllpackDeviceAccessor matrix,
                                    common::Span<GradientPairInt64> d_node_hist,
                                    common::Span<std::uint32_t> d_ridx,
                                    common::Span<GradientQuantiser const> roundings) {
  bst_idx_t n_elements = matrix.row_stride * d_ridx.size();
  for (auto i : dh::GridStrideRange(static_cast<std::size_t>(0), n_elements)) {
    std::uint32_t ridx = i / matrix.row_stride;
    auto fidx = FeatIdx(i, matrix.row_stride);
    auto idx = IterIdx(matrix, ridx, fidx);
    bst_bin_t compressed_bin = matrix.gidx_iter[idx];
    if (compressed_bin == -1) {
      printf("-1\n");
    }
  }
}
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
  auto n = page->Size() * page->info.row_stride;
  auto n_grids = common::DivRoundUp(n, kBlockThreads);

  TestHistBuildKernel<<<n_grids, kBlockThreads>>>(
      std::get<EllpackDeviceAccessor>(page->GetDeviceEllpack(&ctx, {})), node_hist,
      dh::ToSpan(ridx), dh::ToSpan(quantizers));
  debug::SyncDevice();

  // if (use_single_target) {
  //   GradientQuantiser q{GradientPairPrecise{1.0f, 1.0f}, GradientPairPrecise{1.0f, 1.0f}};
  //   histogram.BuildHistogram(ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), fg_acc,
  //                            gpairs.View(ctx.Device()).Values(), dh::ToSpan(ridx), node_hist, q);
  // } else {
  //   auto quantizers = MakeDummyQuantizers(n_targets);
  //   histogram.BuildHistogram(ctx.CUDACtx(), page->GetDeviceEllpack(&ctx, {}), fg_acc,
  //                            gpairs.View(ctx.Device()), dh::ToSpan(ridx), node_hist,
  //                            dh::ToSpan(quantizers));
  // }
}
}  // namespace xgboost::tree::cuda_impl
