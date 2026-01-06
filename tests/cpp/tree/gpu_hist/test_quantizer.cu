#include <gtest/gtest.h>

#include <bitset>
#include <cmath>

#include "../../../../src/tree/gpu_hist/quantiser.cuh"
#include "../../helpers.h"

namespace xgboost::tree {
auto Pf(float v) { return std::bitset<32>{cuda::std::bit_cast<std::uint32_t>(v)}; }
auto Pd(double v) { return std::bitset<64>{cuda::std::bit_cast<std::uint64_t>(v)}; }
auto Pi64(std::int64_t v) { return std::bitset<64>{cuda::std::bit_cast<std::uint64_t>(v)}; }
auto Pi32(std::int32_t v) { return std::bitset<32>{cuda::std::bit_cast<std::uint32_t>(v)}; }
auto Pi16(std::int16_t v) { return std::bitset<16>{cuda::std::bit_cast<std::uint16_t>(v)}; }

XGBOOST_HOST_DEV_INLINE std::uint16_t ExtractExponent(double v) {
  std::uint64_t constexpr kMask = 0x7fff000000000000;
  std::uint64_t iv = *reinterpret_cast<std::uint64_t*>(&v);
  iv &= kMask;
  iv >>= 52;
  return static_cast<std::uint16_t>(iv);
}

XGBOOST_HOST_DEV_INLINE double RestoreExponent(double v, std::uint16_t exponent) {
  std::uint64_t constexpr kMask = ~0x7fff000000000000;
  std::uint64_t iv = cuda::std::bit_cast<std::uint64_t>(v);
  iv &= kMask;
  std::uint64_t res = (std::uint64_t{exponent} << 52) | iv;
  return cuda::std::bit_cast<double>(res);
}

TEST(Quantizer, Extract) {
  // fixed.g:12786314240 h:137438953472 res.g:12786311168 h:137438953472 exp.g:35, h:37
  // to float:2.91038e-11/7.27596e-12, to fixed:3.43597e+10/1.37439e+11
  GradientPairInt64 fixed{12786314240ll, 137438953472ll};
  FixedPointGradScale q{{35, 37}};
  auto f64 = fixed.GetQuantisedGrad() / pow(2.0, 35);
  std::cout << "f64:" << Pd(f64) << std::endl;
  // 0000,0000,0000,0000,0000,0000,0000,001.0,1111,1010,0001,1111,1010,1100,0000,0000
  // 0,01111111101,                        .0,1111,1010,0001,1111,1010,1100,0000,0000,0000,0000,0000,0000,000
  auto exponent = ExtractExponent(f64);
  std::cout << Pi16(exponent) << std::endl;
  double k = f64 + 1.0;
  std::cout << "k  :" << Pd(k) << std::endl;
  auto ve = RestoreExponent(k, exponent);
  std::cout << "res:" << Pd(ve) << std::endl;
}
}  // namespace xgboost::tree
