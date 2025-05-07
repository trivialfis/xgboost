#include <gtest/gtest.h>
#include <thrust/sequence.h>

#include "../../../src/common/cuda_context.cuh"
#include "../../../src/common/cuda_pinned_allocator.h"
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/nvcomp_format.h"
#include "../helpers.h"

namespace xgboost::common {
void test_comp() {
  auto ctx = MakeCUDACtx(0);

  dh::DeviceUVector<CompressedByteT> in(256);
  thrust::sequence(ctx.CUDACtx()->CTP(), in.begin(), in.end(), 0);
  dh::DeviceUVector<std::uint8_t> out;
  CompressEllpack(&ctx, dh::ToSpan(in), &out);

  std::vector<std::uint8_t, cuda_impl::PinnedPoolAllocator<std::uint8_t>> h_in(out.size());
  dh::safe_cuda(cudaMemcpyAsync(h_in.data(), out.data(), out.size() * sizeof(std::uint8_t),
                                cudaMemcpyDefault));
  std::cout << h_in.size() << std::endl;
  std::for_each_n(h_in.begin(), h_in.size(),
                  [](auto v) { std::cout << static_cast<std::int32_t>(v) << ", "; });
  std::cout << std::endl;

  std::cout << "run decom" << std::endl;

  auto h_ptr = h_in.data();
  thrust::for_each_n(
      ctx.CUDACtx()->CTP(), thrust::make_counting_iterator(0ul), h_in.size(),
      [=] XGBOOST_DEVICE(std::size_t i) { printf("%d, ", static_cast<int>(h_ptr[i])); });
  std::cout << "done" << std::endl;
  dh::DebugSyncDevice(__FILE__, __LINE__);
  dh::DeviceUVector<CompressedByteT> dout(in.size());
  DecompressEllpack(curt::DefaultStream(), h_in.data(), dh::ToSpan(dout));
}

TEST(NVComp, Basic) { test_comp(); }
}  // namespace xgboost::common
