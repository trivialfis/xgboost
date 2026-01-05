#include <gtest/gtest.h>

#include <bitset>
#include <cmath>

#include "../../../../src/tree/gpu_hist/quantiser.cuh"
#include "../../helpers.h"

namespace xgboost::tree {
auto Pf(float v) { return std::bitset<32>{cuda::std::bit_cast<std::uint32_t>(v)}; }
auto Pi64(std::int64_t v) { return std::bitset<64>{cuda::std::bit_cast<std::uint64_t>(v)}; }

TEST(Quantizer, FixedPoint) {
  GradientPairPrecise to_fixed{std::pow(2.0, 48), std::pow(2.0, 62)};
  GradientPairPrecise to_float{1.0 / to_fixed.GetGrad(), 1.0 / to_fixed.GetHess()};
  auto q0 = GradientQuantiser{to_fixed, to_float};

  GradientPair one{1.0f, 1.0f};

  // GradientPair g{1.1f, 2.2f};
  // auto fixed = rounding.ToFixedPoint(g);
  // auto gi64 = fixed.GetQuantisedGrad();
  // std::cout << std::hex << gi64 << std::endl;
  // auto hi64 = fixed.GetQuantisedHess();
  // std::cout << std::hex << hi64 << std::endl;
  // 48, 52
  // 1000000000000
  // 20000000000000
  // 16, 17
  // 10000
  // 40000
  // 16/1.1, 17/1.2
  // 11999
  // 46666
  // 48/1.1, 52/1.2
  //    1,1999,9a00,0000
  //   23,3333,4000,0000
  // full-64
  // 0000,0000,0000,0000
  // full-32
  // 0000,0000

  // std::locale loc("");  // system default locale
  // std::cout.imbue(loc);
  using cuda::std::bit_cast;
  auto ctx = MakeCUDACtx(0);
  auto max_g =
      GradientPair{std::numeric_limits<float>::max(), std::numeric_limits<float>::epsilon()};
  std::cout << "eps:" << max_g.GetHess() << "\n"
            << std::bitset<32>{cuda::std::bit_cast<std::uint32_t>(max_g.GetHess())} << std::endl;
  auto v = std::stoi("00100000000000000000000000000000", nullptr, 2);
  std::cout << v << ", f:" << bit_cast<float>(v) << "\n"
            << "one:" << std::bitset<32>{bit_cast<std::uint32_t>(1.0f)} << std::endl;
  v = std::stoi("00111111100000000000000000000001", nullptr, 2);
  auto v_fixed = q0.ToFixedPoint(GradientPair{bit_cast<float>(v), bit_cast<float>(v)});
  // 62 bits for the fractional part,
  //  |--,----,----,----,----,----,-|  23-bit mantissa
  // 0100,0000,0000,0000,0000,0000,1000,0000,0000,0000,0000,0000,0000,0000,0000,0000
  std::cout << "v:" << v << ", f:" << bit_cast<float>(v) << "\n"
            << Pi64(v_fixed.GetQuantisedHess()) << std::endl;

  std::cout << max_g << std::endl;
  // 0,01111111,00000000000000000000000
  dh::device_vector<GradientPair> values(512, max_g);
  MetaInfo info;
  info.num_row_ = values.size();
  auto q = GradientQuantiser{&ctx, linalg::MakeVec(ctx.Device(), dh::ToSpan(values)), info};
  // auto esp = std::numeric_limits<float>::epsilon();
  auto fixed_max_g = q.ToFixedPoint(max_g);
  std::cout << fixed_max_g.GetQuantisedGrad()
            << " f64:" << static_cast<double>(fixed_max_g.GetQuantisedGrad()) << std::endl;
  // std::cout <<
  // std::bitset<64>{cuda::std::bit_cast<std::uint64_t>(fixed_max_g.GetQuantisedGrad())}
  //           << " f64:" << static_cast<double>(fixed_max_g.GetQuantisedGrad()) << std::endl;
  // auto one_fixed = q.ToFixedPoint(one);
  // std::cout << "one:"
  //           << std::bitset<64>{cuda::std::bit_cast<std::uint64_t>(one_fixed.GetQuantisedHess())}
  //           << " f64:" << static_cast<double>(one_fixed.GetQuantisedHess()) << std::endl;
  // 0000,0000,0001,1111,1111,1111,1111,1111,1110,0000,0000,0000,0000,0000,0000,0000
  std::cout << fixed_max_g.GetQuantisedHess() << "\n"
            << std::bitset<64>{cuda::std::bit_cast<std::uint64_t>(fixed_max_g.GetQuantisedGrad())}
            << std::endl;
  // 1f,ffff,e000,0000
  // 1f,ffff,e000,0000
  std::cout << q.ToFloatingPoint(fixed_max_g).GetGrad() << std::endl;
  std::cout << q.ToFloatingPoint(fixed_max_g).GetHess() << std::endl;
  auto max_i32 = std::numeric_limits<std::int32_t>::max();
  std::cout << "max i32:" << static_cast<double>(max_i32) << " i:" << max_i32 << std::endl;
  auto max_i64 = std::numeric_limits<std::int64_t>::max();
  std::cout << "max i64:" << static_cast<double>(max_i64) << " i:" << max_i64 << std::endl;
  // 9223372036854775807: max i64
  //    9007198717870080
}

TEST(Quantizer, Controlled) {
  using cuda::std::bit_cast;

  GradientPairPrecise to_fixed{std::pow(2.0, 75), std::pow(2.0, -75)};
  GradientPairPrecise to_float{1.0 / to_fixed.GetGrad(), 1.0 / to_fixed.GetHess()};

  auto q0 = GradientQuantiser{to_fixed, to_float};

  auto v = std::stoi("00111111100000000000000000000001", nullptr, 2);
  auto v_fixed = q0.ToFixedPoint(GradientPair{bit_cast<float>(v), bit_cast<float>(v)});
  std::cout << "v:" << v << ", f:" << bit_cast<float>(v) << "\n"
            << Pi64(v_fixed.GetQuantisedGrad()) << "\n"
            << Pi64(v_fixed.GetQuantisedHess()) << std::endl;
}
}  // namespace xgboost::tree
