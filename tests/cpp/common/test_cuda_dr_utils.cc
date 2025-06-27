/**
 * Copyright 2025, XGBoost Contributors
 */
#include <gtest/gtest.h>

#if defined(XGBOOST_USE_CUDA) && defined(__linux__)
#include <unordered_set>

#include "../../../src/common/common.h"
#include "../../../src/common/cuda_dr_utils.h"

namespace xgboost::cudr {
TEST(DrUtils, GetVersionFromSmi) {
  std::int32_t major = 0, minor = 0;
  bool result = GetVersionFromSmi(&major, &minor);

  if (result) {
    EXPECT_GE(major, 0);
    EXPECT_GE(minor, 0);
  } else {
    EXPECT_EQ(major, -1);
    EXPECT_EQ(minor, -1);
  }
}

TEST(DrUtils, GetC2cLinkCountFromSmi) {
  {
    auto out = R"(GPU 0: NVIDIA GH200 480GB (UUID: GPU-********-****-****-****-************)
    C2C Link 0: 44.712 GB/s
    C2C Link 1: 44.712 GB/s
    C2C Link 2: 44.712 GB/s
    C2C Link 3: 44.712 GB/s
    C2C Link 4: 44.712 GB/s
    C2C Link 5: 44.712 GB/s
    C2C Link 6: 44.712 GB/s
    C2C Link 7: 44.712 GB/s
    C2C Link 8: 44.712 GB/s
    C2C Link 9: 44.712 GB/s
  )";
    auto lc = detail::GetC2cLinkCountFromSmiImpl(out);
    ASSERT_EQ(lc, 10);
  }
  {
    auto out = R"(No Devices support C2C.
)";
    auto lc = detail::GetC2cLinkCountFromSmiImpl(out);
    ASSERT_EQ(lc, -1);
  }

  {
    [[maybe_unused]] auto _ = GetC2cLinkCountFromSmi();
  }
  {
    [[maybe_unused]] auto _ = GetC2cLinkCountFromSmiGlobal();
  }
}

namespace {
std::string GetCuDeviceUuid(std::int32_t ordinal) {
  CUuuid dev_uuid;
  std::stringstream s;
  std::unordered_set<unsigned char> dashPos{0, 4, 6, 8, 10};
  cudr::GetGlobalCuDriverApi().cuDeviceGetUuid(&dev_uuid, ordinal);

  s << "GPU";
  for (int i = 0; i < 16; i++) {
    if (dashPos.count(i)) {
      s << '-';
    }
    s << std::hex << std::setfill('0') << std::setw(2)
      << (0xFF & static_cast<std::int32_t>(dev_uuid.bytes[i]));
  }
  return s.str();
}

std::string GetCudaUUID(std::int32_t ordinal) {
  cudaDeviceProp prob{};
  dh::safe_cuda(cudaGetDeviceProperties(&prob, ordinal));

  std::unordered_set<unsigned char> dashPos{0, 4, 6, 8, 10};
  std::stringstream s;
  s << "GPU";
  for (int i = 0; i < 16; i++) {
    if (dashPos.count(i)) {
      s << '-';
    }
    s << std::hex << std::setfill('0') << std::setw(2)
      << (0xFF & static_cast<std::int32_t>(prob.uuid.bytes[i]));
  }

  return s.str();
}
}  // namespace

TEST(DrUtils, GetUUID) {
  [[maybe_unused]] auto& _ = cudr::GetGlobalCuDriverApi();
  std::cout << "Driver" << std::endl;
  for (std::size_t i = 0; i < 2; ++i) {
    std::cout << "i:" << i << " " << GetCuDeviceUuid(i) << std::endl;
  }
  std::cout << "Runtime" << std::endl;
  for (std::size_t i = 0; i < 2; ++i) {
    std::cout << "i:" << i << " " << GetCudaUUID(i) << std::endl;
  }
}
}  // namespace xgboost::cudr
#endif  // defined(XGBOOST_USE_CUDA)
