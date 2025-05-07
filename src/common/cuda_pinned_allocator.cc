/**
 * Copyright 2025, XGBoost Contributors
 */
#include "cuda_pinned_allocator.h"

#include "common.h"

namespace xgboost::common::cuda_impl {
cudaMemPool_t* CreateHostMemPool() {
  auto del = [](cudaMemPool_t* mem_pool) {
    if (mem_pool) {
      dh::safe_cuda(cudaMemPoolDestroy(*mem_pool));
    }
  };

  static std::unique_ptr<cudaMemPool_t, void (*)(cudaMemPool_t*)> mem_pool{nullptr, del};
  static std::once_flag once;
  std::call_once(once, [] {
    mem_pool = std::unique_ptr<cudaMemPool_t, void (*)(cudaMemPool_t*)>{
        [] {
          cudaMemPoolProps props = {};
          props.location.type = cudaMemLocationTypeHostNuma;
          props.location.id = 0;
          props.allocType = cudaMemAllocationTypePinned;
          props.usage = cudaMemPoolCreateUsageHwDecompress;

          cudaMemPool_t* mem_pool = new cudaMemPool_t;
          dh::safe_cuda(cudaMemPoolCreate(mem_pool, &props));
          return mem_pool;
        }(),
        [](cudaMemPool_t* mem_pool) {
          if (mem_pool) {
            dh::safe_cuda(cudaMemPoolDestroy(*mem_pool));
          }
        }};
  });

  CHECK(mem_pool);
  return mem_pool.get();
}
}  // namespace xgboost::common::cuda_impl
