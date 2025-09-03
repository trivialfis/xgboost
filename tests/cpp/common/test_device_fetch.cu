#include <gtest/gtest.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <xgboost/span.h>

#include <cuda/pipeline>

#include "../../../src/common/common.h"

namespace dh {
namespace {
int constexpr static kItemsPerThread = 8;
int constexpr static kBlockThreads = 1024;
int constexpr static kItemsPerTile = kBlockThreads * kItemsPerThread;
}  // namespace

__global__ void TestPrefetchKernel(float const* ptr, std::size_t n_elements, float* out) {
  extern __shared__ char smem[];
  auto stage_mem = reinterpret_cast<float*>(smem);

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  constexpr int kStages = 2;
  int constexpr kStageSize = kItemsPerThread / kStages;

  // Block offset, beginning index of the block. Individual thread needs to add threadIdx
  std::size_t offset = blockIdx.x * kItemsPerTile;
  constexpr int kItemSize = sizeof(float);

  auto load = [&](std::size_t offset, int stage) {
#pragma unroll
    for (int i = 0; i < kStageSize; i++) {
      auto k = stage * kStageSize + i;
      auto idx = offset + k * kBlockThreads + threadIdx.x;
      auto shmem_beg_idx = kBlockThreads * i + (threadIdx.x);
      shmem_beg_idx = shmem_beg_idx + stage * kBlockThreads * kStageSize;
      cuda::memcpy_async(stage_mem + shmem_beg_idx, ptr + idx, kItemSize, pipe);
    }
  };
  auto comp = [&](std::size_t offset, int stage) {
#pragma unroll
    for (int i = 0; i < kStageSize; i++) {
      auto shmem_beg_idx = kBlockThreads * i + (threadIdx.x);
      shmem_beg_idx = shmem_beg_idx + stage * kBlockThreads * kStageSize;

      auto k = stage * kStageSize + i;
      auto idx = offset + k * kBlockThreads + threadIdx.x;

      cuda::std::memcpy(out + idx, stage_mem + shmem_beg_idx, kItemSize);
    }
  };
  auto partial_comp = [] {
  };

  auto stage = 0;

  auto flip_stage = [&] {
    stage = (stage + 1) % kStages;
  };

  pipe.producer_acquire();
  if (offset + kItemsPerTile <= n_elements) {
    load(offset, stage);
  }
  pipe.producer_commit();

  flip_stage();  // s -> 1

  pipe.producer_acquire();
  if (offset + kItemsPerTile <= n_elements) {
    load(offset, stage);
  }
  pipe.producer_commit();

  flip_stage();  // s -> 0

  // kItemsPerTile * gridDim.x  =>  blockDim.x * kItemsPerThread * gridDim.x
  // grid strided range
  std::size_t c_offset = offset;

  while (c_offset + kItemsPerTile <= n_elements) {
    // Consume
    cuda::pipeline_consumer_wait_prior<1>(pipe);
    comp(c_offset, stage);
    pipe.consumer_release();

    // Re-fill
    pipe.producer_acquire();
    c_offset = offset;
    offset += (kItemsPerTile * gridDim.x) * ((stage + 1) % 2);
    if (offset + kItemsPerTile <= n_elements) {
      load(offset, stage);
    }

    flip_stage();
    pipe.producer_commit();
  }
  partial_comp();
}

void TestPrefetch() {
  std::size_t n = 1024 * 1024 * 1024;
  float *ptr = nullptr, *out = nullptr;
  dh::safe_cuda(cudaMalloc(&ptr, sizeof(float) * n));
  dh::safe_cuda(cudaMalloc(&out, sizeof(float) * n));

  thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0ul), n,
                     [=] __device__(std::size_t i) { ptr[i] = i; });
  dim3 const block_dim{kBlockThreads};
  std::size_t shmem = kItemsPerThread * kBlockThreads * sizeof(float);
  dim3 const grid_dim{512};
  // ASSERT_EQ(grid_dim.x, 16384);
  TestPrefetchKernel<<<grid_dim, block_dim, shmem>>>(ptr, n, out);

  std::vector<float> h_out(n, 0);
  safe_cuda(cudaMemcpy(h_out.data(), out, sizeof(float) * h_out.size(), cudaMemcpyDefault));

  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_EQ(h_out[i], i);
  }

  safe_cuda(cudaFree(ptr));
  safe_cuda(cudaFree(out));
}

TEST(DevicePrefetch, Loop) { TestPrefetch(); }
}  // namespace dh
