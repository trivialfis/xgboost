#include <gtest/gtest.h>

#include <cuda/pipeline>

#include "../../../src/common/common.h"

namespace xgboost::dh {
namespace {
int constexpr static kItemsPerThread = 8;
int constexpr static kBlockThreads = 1024;
int constexpr static kItemsPerTile = kBlockThreads * kItemsPerThread;
}  // namespace

__global__ void TestPrefetchKernel(std::size_t n_elements) {
  extern __shared__ char smem[];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  constexpr int kStages = 2;
  int constexpr kStageSize = kItemsPerThread / kStages;

  // Block offset, beginning index of the block. Individual thread needs to add threadIdx
  std::size_t offset = blockIdx.x * kItemsPerTile;

  auto load = [] (std::size_t offset){
  };
  auto comp = [] {
  };
  auto partial_comp = [] {
  };

  auto stage = 0;

  auto flip_stage = [&] {
    stage = (stage + 1) % kStages;
  };

  pipe.producer_acquire();
  load(offset);
  pipe.producer_commit();

  flip_stage();  // s -> 1

  pipe.producer_acquire();
  load(offset);
  pipe.producer_commit();

  flip_stage();  // s -> 0

  // kItemsPerTile * gridDim.x -> blockDim.x * kItemsPerThread * gridDim.x
  // grid strided range
  // for (std::size_t block = offset;)
  while (offset + kItemsPerTile <= n_elements) {
    // Consume
    cuda::pipeline_consumer_wait_prior<1>(pipe);
    comp();
    pipe.consumer_release();

    // Re-fill
    pipe.producer_acquire();
    offset += (kItemsPerTile * gridDim.x) * ((stage + 1) % 2);
    load(offset);
    flip_stage();
    pipe.producer_commit();
  }

  partial_comp();
}

TEST(DevicePrefetch, Loop) {
  std::uint32_t n = 4096;
  dim3 const block_dim{kBlockThreads};
  dim3 const grid_dim{common::DivRoundUp(n, kBlockThreads)};
  TestPrefetchKernel<<<grid_dim, block_dim>>>(n);
}
}  // namespace xgboost::dh
