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
          std::cout << "create mem pool" << std::endl;

          cudaMemPoolProps props;
          std::memset(&props, '\0', sizeof(props));
          props.location.type = cudaMemLocationTypeHostNuma;
          std::int32_t numa_id = -1;
          dh::safe_cuda(cudaDeviceGetAttribute(&numa_id, cudaDevAttrNumaId, curt::CurrentDevice()));
          numa_id = std::max(numa_id, 0);
          props.location.id = numa_id;
          props.allocType = cudaMemAllocationTypePinned;
          props.usage = cudaMemPoolCreateUsageHwDecompress;
          props.handleTypes = cudaMemHandleTypeNone;

          cudaMemPoolProps dprops;
          std::memset(&dprops, '\0', sizeof(dprops));
          dprops.location.type = cudaMemLocationTypeDevice;
          dprops.location.id = curt::CurrentDevice();
          // dprops.allocType = cudaMemAllocationTypePinned;
          dprops.usage = cudaMemPoolCreateUsageHwDecompress;
          dprops.handleTypes = cudaMemHandleTypeNone;

          std::vector<cudaMemPoolProps> vprops{props, dprops};

          cudaMemPool_t* mem_pool = new cudaMemPool_t;
          dh::safe_cuda(cudaMemPoolCreate(mem_pool, vprops.data()));
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
