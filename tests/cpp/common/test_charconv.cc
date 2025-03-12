/*
 * The code is adopted from original (half) c implementation:
 * https://github.com/ulfjack/ryu.git with some more comments and tidying.  License is
 * attached below.
 *
 * Copyright 2018 Ulf Adams
 *
 * The contents of this file may be used under the terms of the Apache License,
 * Version 2.0.
 *
 *    (See accompanying file LICENSE-Apache or copy at
 *     http: *www.apache.org/licenses/LICENSE-2.0)
 *
 * Alternatively, the contents of this file may be used under the terms of
 * the Boost Software License, Version 1.0.
 *    (See accompanying file LICENSE-Boost or copy at
 *     https://www.boost.org/LICENSE_1_0.txt)
 *
 * Unless required by applicable law or agreed to in writing, this software
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>  // for Context

#include <cstddef>
#include <limits>

#include "../../../src/common/charconv.h"
#include "../../../src/common/threading_utils.h"  // for ParallelFor

namespace xgboost {
namespace {
void TestInteger(char const* res, int64_t i) {
  char result[xgboost::NumericLimits<int64_t>::kToCharsSize];
  auto ret = to_chars(result, result + sizeof(result), i);
  *ret.ptr = '\0';
  EXPECT_STREQ(res, result);
}

static float Int32Bits2Float(uint32_t bits) {
  float f;
  memcpy(&f, &bits, sizeof(float));
  return f;
}

void TestRyu(char const *res, float v) {
  char result[xgboost::NumericLimits<float>::kToCharsSize];
  auto ret = to_chars(result, result + sizeof(result), v);
  *ret.ptr = '\0';
  EXPECT_STREQ(res, result);
}
}  // anonymous namespace

TEST(Ryu, Subnormal) {
  TestRyu("0E0", 0.0f);
  TestRyu("-0E0", -0.0f);
  TestRyu("1E0", 1.0f);
  TestRyu("-1E0", -1.0f);
  TestRyu("NaN", NAN);
  TestRyu("Infinity", INFINITY);
  TestRyu("-Infinity", -INFINITY);

  TestRyu("1E-45", std::numeric_limits<float>::denorm_min());
}

TEST(Ryu, Denormal) {
  TestRyu("1E-45", std::numeric_limits<float>::denorm_min());
}

TEST(Ryu, SwitchToSubnormal) {
  TestRyu("1.1754944E-38", 1.1754944E-38f);
}

TEST(Ryu, MinAndMax) {
  TestRyu("3.4028235E38", Int32Bits2Float(0x7f7fffff));
  TestRyu("1E-45", Int32Bits2Float(1));
}

// Check that we return the exact boundary if it is the shortest
// representation, but only if the original floating point number is even.
TEST(Ryu, BoundaryRoundEven) {
  TestRyu("3.355445E7", 3.355445E7f);
  TestRyu("9E9", 8.999999E9f);
  TestRyu("3.436672E10", 3.4366717E10f);
}

// If the exact value is exactly halfway between two shortest representations,
// then we round to even. It seems like this only makes a difference if the
// last two digits are ...2|5 or ...7|5, and we cut off the 5.
TEST(Ryu, ExactValueRoundEven) {
  TestRyu("3.0540412E5", 3.0540412E5f);
  TestRyu("8.0990312E3", 8.0990312E3f);
}

TEST(Ryu, LotsOfTrailingZeros) {
  // Pattern for the first test: 00111001100000000000000000000000
  TestRyu("2.4414062E-4", 2.4414062E-4f);
  TestRyu("2.4414062E-3", 2.4414062E-3f);
  TestRyu("4.3945312E-3", 4.3945312E-3f);
  TestRyu("6.3476562E-3", 6.3476562E-3f);
}

TEST(Ryu, Regression) {
  TestRyu("4.7223665E21", 4.7223665E21f);
  TestRyu("8.388608E6", 8388608.0f);
  TestRyu("1.6777216E7", 1.6777216E7f);
  TestRyu("3.3554436E7", 3.3554436E7f);
  TestRyu("6.7131496E7", 6.7131496E7f);
  TestRyu("1.9310392E-38", 1.9310392E-38f);
  TestRyu("-2.47E-43", -2.47E-43f);
  TestRyu("1.993244E-38", 1.993244E-38f);
  TestRyu("4.1039004E3", 4103.9003f);
  TestRyu("5.3399997E9", 5.3399997E9f);
  TestRyu("6.0898E-39", 6.0898E-39f);
  TestRyu("1.0310042E-3", 0.0010310042f);
  TestRyu("2.882326E17", 2.8823261E17f);
  TestRyu("7.038531E-26", 7.0385309E-26f);
  TestRyu("9.223404E17", 9.2234038E17f);
  TestRyu("6.710887E7", 6.7108872E7f);
  TestRyu("1E-44", 1.0E-44f);
  TestRyu("2.816025E14", 2.816025E14f);
  TestRyu("9.223372E18", 9.223372E18f);
  TestRyu("1.5846086E29", 1.5846085E29f);
  TestRyu("1.1811161E19", 1.1811161E19f);
  TestRyu("5.368709E18", 5.368709E18f);
  TestRyu("4.6143166E18", 4.6143165E18f);
  TestRyu("7.812537E-3", 0.007812537f);
  TestRyu("1E-45", 1.4E-45f);
  TestRyu("1.18697725E20", 1.18697724E20f);
  TestRyu("1.00014165E-36", 1.00014165E-36f);
  TestRyu("2E2", 200.0f);
  TestRyu("3.3554432E7", 3.3554432E7f);

  static_assert(1.1920929E-7f == std::numeric_limits<float>::epsilon());
  TestRyu("1.1920929E-7", std::numeric_limits<float>::epsilon());
}

TEST(Ryu, RoundTrip) {
  float f = -1.1493590134238582e-40;
  char result[NumericLimits<float>::kToCharsSize] { 0 };
  auto ret = to_chars(result, result + sizeof(result), f);
  size_t dis = std::distance(result, ret.ptr);
  float back;
  auto from_ret = from_chars(result, result + dis, back);
  ASSERT_EQ(from_ret.ec, std::errc());
  std::string str;
  for (size_t i = 0; i < dis; ++i) {
    str.push_back(result[i]);
  }
  ASSERT_EQ(f, back);
}

TEST(Ryu, LooksLikePow5) {
  // These numbers have a mantissa that is the largest power of 5 that fits,
  // and an exponent that causes the computation for q to result in 10, which is a corner
  // case for Ryu.
  TestRyu("6.7108864E17", Int32Bits2Float(0x5D1502F9));
  TestRyu("1.3421773E18", Int32Bits2Float(0x5D9502F9));
  TestRyu("2.6843546E18", Int32Bits2Float(0x5E1502F9));
}

TEST(Ryu, OutputLength) {
  TestRyu("1E0", 1.0f); // already tested in Basic
  TestRyu("1.2E0", 1.2f);
  TestRyu("1.23E0", 1.23f);
  TestRyu("1.234E0", 1.234f);
  TestRyu("1.2345E0", 1.2345f);
  TestRyu("1.23456E0", 1.23456f);
  TestRyu("1.234567E0", 1.234567f);
  TestRyu("1.2345678E0", 1.2345678f);
  TestRyu("1.23456735E-36", 1.23456735E-36f);
}

TEST(IntegerPrinting, Basic) {
  TestInteger("0", 0);
  auto str = std::to_string(std::numeric_limits<int64_t>::min());
  TestInteger(str.c_str(), std::numeric_limits<int64_t>::min());
  str = std::to_string(std::numeric_limits<int64_t>::max());
  TestInteger(str.c_str(), std::numeric_limits<int64_t>::max());
}

void TestRyuParse(float f, std::string in) {
  float res;
  auto ret = from_chars(in.c_str(), in.c_str() + in.size(), res);
  ASSERT_EQ(ret.ec, std::errc());
  ASSERT_EQ(f, res);
}

TEST(Ryu, Basic) {
  TestRyuParse(0.0f, "0");
  TestRyuParse(-0.0f, "-0");
  TestRyuParse(1.0f, "1");
  TestRyuParse(-1.0f, "-1");
  TestRyuParse(123456792.0f, "123456789");
  TestRyuParse(299792448.0f, "299792458");
}

TEST(Ryu, MinMax) {
  TestRyuParse(1e-45f, "1e-45");
  TestRyuParse(FLT_MIN, "1.1754944e-38");
  TestRyuParse(FLT_MAX, "3.4028235e+38");
}

TEST(Ryu, MantissaRoundingOverflow) {
  TestRyuParse(1.0f, "0.999999999");
  TestRyuParse(INFINITY, "3.4028236e+38");
  TestRyuParse(1.1754944e-38f, "1.17549430e-38"); // FLT_MIN
}

TEST(Ryu, TrailingZeros) {
  TestRyuParse(26843550.0f, "26843549.5");
  TestRyuParse(50000004.0f, "50000002.5");
  TestRyuParse(99999992.0f, "99999989.5");
}

void TestRyuParseF64(double f, std::string in) {
  double res{std::numeric_limits<double>::quiet_NaN()};
  char buf[NumericLimits<double>::kToCharsSize];
  std::memset(buf, '\0', NumericLimits<double>::kToCharsSize);
  std::copy_n(in.data(), in.size(), buf);
  auto ret = from_chars(buf, buf + in.size(), res);
  ASSERT_EQ(ret.ec, std::errc{});
  ASSERT_EQ(f, res);
}

TEST(Ryu, BasicF64) {
  TestRyuParseF64(0.0, "0");
  TestRyuParseF64(-0.0, "-0");
  TestRyuParseF64(1.0, "1");
  TestRyuParseF64(2.0, "2");
  TestRyuParseF64(123456789.0, "123456789");
  TestRyuParseF64(123.456, "123.456");
  TestRyuParseF64(123.456, "123456e-3");
  TestRyuParseF64(123.456, "1234.56e-1");
  TestRyuParseF64(1.453, "1.453");
  TestRyuParseF64(1453.0, "1.453e+3");
  TestRyuParseF64(0.0, ".0");
  TestRyuParseF64(1.0, "1e0");
  TestRyuParseF64(1.0, "1E0");
  TestRyuParseF64(1.0, "000001.000000");
  TestRyuParseF64(0.2316419, "0.2316419");
}

TEST(Ryu, MinMaxF64) {
  TestRyuParseF64(1.7976931348623157e308, "1.7976931348623157e308");
  TestRyuParseF64(5E-324, "5E-324");
}

TEST(Ryu, MantissaRoundingOverflowF64) {
  // This results in binary mantissa that is all ones and requires rounding up
  // because it is closer to 1 than to the next smaller float. This is a
  // regression test that the mantissa overflow is handled correctly by
  // increasing the exponent.
  TestRyuParseF64(1.0, "0.99999999999999999");
  // This number overflows the mantissa *and* the IEEE exponent.
  TestRyuParseF64(INFINITY, "1.7976931348623159e308");
}

TEST(Ryu, UnderflowF64) {
  TestRyuParseF64(0.0, "2.4e-324");
  TestRyuParseF64(0.0, "1e-324");
  TestRyuParseF64(0.0, "9.99999e-325");
  // These are just about halfway between 0 and the smallest float.
  // The first is just below the halfway point, the second just above.
  TestRyuParseF64(0.0, "2.4703282292062327e-324");
  TestRyuParseF64(5e-324, "2.4703282292062328e-324");
}

TEST(Ryu, OverflowF64) {
  TestRyuParseF64(INFINITY, "2e308");
  TestRyuParseF64(INFINITY, "1e309");
}

TEST(Ryu, TableSizeDenormalF64) {
  TestRyuParseF64(5e-324, "4.9406564584124654e-324");
}

TEST(Ryu, SLOW_Double) {
  auto test = [](std::uint64_t i) {
    double f;
    std::memcpy(&f, &i, sizeof(f));
    char buf[NumericLimits<double>::kToCharsSize];
    auto res = to_chars(buf, buf + NumericLimits<double>::kToCharsSize, f);
    *res.ptr = '\0';
    ASSERT_EQ(res.ec, std::errc{});

    double f1;
    auto res1 = from_chars(buf, buf + NumericLimits<double>::kToCharsSize, f1);
    ASSERT_EQ(res1.ec, std::errc{});

    switch (std::fpclassify(f)) {
      case FP_INFINITE: {
        ASSERT_TRUE(std::isinf(f1));
        break;
      }
      case FP_NAN: {
        ASSERT_TRUE(std::isnan(f1));
        break;
      }
      case FP_SUBNORMAL:
        // Underflow, rounding.
        break;
      default: {
        ASSERT_EQ(f, f1);
      }
    }
  };
  std::uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
  Context ctx;
  common::ParallelFor(uint64_max, ctx.Threads(), [&](auto i) { test(static_cast<uint32_t>(i)); });
}
}  // namespace xgboost
