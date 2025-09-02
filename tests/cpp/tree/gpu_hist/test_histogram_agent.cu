/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <thrust/sequence.h>

#include "../../../../src/common/type.h"
#include "../../../../src/data/ellpack_page.cuh"
#include "../../../../src/tree/gpu_hist/histogram_agent.cuh"
#include "../../helpers.h"
namespace xgboost::tree::cuda_impl {
using Idx = std::uint32_t;
template <typename Accessor>
__global__ void TestWriteBack(Accessor acc, FeatureGroupsAccessor groups,
                              common::Span<Idx const> d_ridx, GradientQuantiser const& rounding,
                              const GradientPair* d_gpair, common::Span<std::uint32_t> d_out) {
  extern __shared__ char smem[];
  const FeatureGroup group = groups[blockIdx.y];
  HistogramAgent<common::GetValueT<decltype(acc)>, true, true, 1024, 8> agent{
      reinterpret_cast<GradientPairInt64*>(smem), nullptr, group, acc, d_ridx, rounding, d_gpair};
  agent.BuildHistogramWithShared([&](bst_bin_t dst, auto adjusted) { atomicAdd(&d_out[dst], 1); },
                                 [](auto, auto) {

                                 });
}

void TestHistAgentLoad() {
  // Build data
  bst_idx_t n_samples = 4096, n_features = 8;
  bst_bin_t n_bins = 4;
  auto ctx = MakeCUDACtx(0);
  auto p_fmat = RandomDataGenerator{n_samples, n_features, 0.0}
                    .Device(ctx.Device())
                    .Bins(n_bins)
                    .GenerateQuantileDMatrix(false);
  auto it = p_fmat->GetBatches<EllpackPage>(&ctx, BatchParam{n_bins, 0.2}).begin();
  auto const& page = *it;
  auto gpair = GenerateRandomGradients(&ctx, n_samples, 1);
  // Create a callback to write back the loaded gidx.

  // Pass the callback to BuildHistogramWithShared.
  auto max_shmem = dh::MaxSharedMemoryOptin(ctx.Ordinal());
  FeatureGroups groups{*page.Impl()->CutsShared(), true, max_shmem};

  dh::device_vector<Idx> ridx(n_samples);
  thrust::sequence(ctx.CUDACtx()->CTP(), ridx.begin(), ridx.end(), 0);
  auto d_ridx = dh::ToSpan(ridx);
  auto d_gpair = gpair.View(ctx.Device()).Values();
  GradientQuantiser quantiser{&ctx, d_gpair, p_fmat->Info()};

  HostDeviceVector<std::uint32_t> out(n_features * n_bins, 0, ctx.Device());
  auto d_out = out.DeviceSpan();

  page.Impl()->Visit(&ctx, {}, [&](auto&& acc) {
    constexpr int kBlockThreads = 1024;
    using Accessor = common::GetValueT<decltype(acc)>;
    auto d_groups = groups.DeviceAccessor(ctx.Device());
    auto smem_size = max_shmem;  // fixme: actual size
    auto kernel = TestWriteBack<Accessor>;
    dh::safe_cuda(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem));

    std::int32_t num_groups = d_groups.NumGroups();
    std::int32_t n_mps = 0;
    dh::safe_cuda(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, ctx.Ordinal()));

    std::int32_t n_blocks_per_mp = 0;
    dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n_blocks_per_mp, kernel,
                                                                kBlockThreads, smem_size));

    auto grid_size = n_blocks_per_mp * n_mps;

    dh::LaunchKernel{dim3(grid_size, d_groups.NumGroups()),  // NOLINT
                     static_cast<uint32_t>(kBlockThreads), smem_size,
                     ctx.CUDACtx()->Stream()}(TestWriteBack<common::GetValueT<decltype(acc)>>, acc,
                                              d_groups, d_ridx, quantiser, d_gpair.data(), d_out);
  });
  ctx.CUDACtx()->Stream().Sync();
  // Check the gidx.
  auto const& h_out = out.ConstHostVector();
  std::vector<std::uint32_t> exp{1024, 1024, 1023, 1025, 1024, 1024, 1023, 1025, 1024, 1024, 1023,
                                 1025, 1024, 1024, 1023, 1025, 1024, 1024, 1023, 1025, 1024, 1024,
                                 1023, 1025, 1024, 1024, 1023, 1025, 1024, 1024, 1023, 1025};
  ASSERT_EQ(exp, h_out);
}

TEST(HistogramAgent, Load) { TestHistAgentLoad(); }
}  // namespace xgboost::tree::cuda_impl
