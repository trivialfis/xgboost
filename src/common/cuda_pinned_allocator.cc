/**
 * Copyright 2025, XGBoost Contributors
 */
#include "cuda_pinned_allocator.h"

#include "common.h"
#include "cuda_rt_utils.h"

namespace xgboost::common::cuda_impl {
/**
cudaHostAlloc:	0
cudaMallocHost:	0
cudaHostRegister:	0
cudaMalloc:	1
cudaMallocManaged:	0
mem_pool_ptr:	1
minimum granularity:	2097152
mem_create_ptr:	1
 */
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
          curt::DefaultStream().Sync();
          std::cout << "create mem pool" << std::endl;

          // setting location fails on gb for some reason. both device and host.

          cudaMemPoolProps props;
          std::memset(&props, '\0', sizeof(props));
          props.location.type = cudaMemLocationTypeHostNuma;
          std::int32_t numa_id = -1;
          dh::safe_cuda(cudaDeviceGetAttribute(&numa_id, cudaDevAttrNumaId, curt::CurrentDevice()));
          numa_id = std::max(numa_id, 0);
          std::cout << "numa_id: " << numa_id << std::endl;
          // props.location.id = numa_id;
          props.allocType = cudaMemAllocationTypePinned;
          props.usage = cudaMemPoolCreateUsageHwDecompress;
          props.handleTypes = cudaMemHandleTypeNone;

          cudaMemPoolProps dprops;
          std::memset(&dprops, '\0', sizeof(dprops));
          dprops.location.type = cudaMemLocationTypeDevice;
          // dprops.location.id = 0;
          dprops.allocType = cudaMemAllocationTypePinned;
          dprops.usage = cudaMemPoolCreateUsageHwDecompress;
          dprops.handleTypes = cudaMemHandleTypeNone;

          std::vector<cudaMemPoolProps> vprops{props, dprops};

          cudaMemPool_t* mem_pool = new cudaMemPool_t;
          dh::safe_cuda(cudaMemPoolCreate(mem_pool, vprops.data()));
          std::cout << "created" << std::endl;

          cudaMemAccessDesc h_desc;
          h_desc.location.type = cudaMemLocationTypeHostNuma;
          h_desc.location.id = 0;
          h_desc.flags = cudaMemAccessFlagsProtReadWrite;

          cudaMemAccessDesc d_desc;
          d_desc.location.type = cudaMemLocationTypeDevice;
          d_desc.location.id = 0;
          d_desc.flags = cudaMemAccessFlagsProtReadWrite;

          std::vector<cudaMemAccessDesc> descs{h_desc, d_desc};
          dh::safe_cuda(cudaMemPoolSetAccess(*mem_pool, descs.data(), descs.size()));
          return mem_pool;
        }(),
        [](cudaMemPool_t* mem_pool) {
          if (mem_pool) {
            std::cout << "delete pool" << std::endl;
            dh::safe_cuda(cudaMemPoolDestroy(*mem_pool));
          }
        }};
  });

  CHECK(mem_pool);
  return mem_pool.get();
}
}  // namespace xgboost::common::cuda_impl
