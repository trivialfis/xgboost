/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#pragma once
#include "../../common/device_helpers.cuh"  // for ToSpan
#include "../../common/device_vector.cuh"   // for device_vector
#include "xgboost/base.h"                   // for GradientPairPrecise, GradientPairInt64
#include "xgboost/context.h"                // for Context
#include "xgboost/data.h"                   // for MetaInfo
#include "xgboost/linalg.h"                 // for VectorView

namespace xgboost::tree {
namespace detail {
inline std::int32_t constexpr kF32MantissaBits = 23;

XGBOOST_DEVICE inline std::int32_t ExtractFixed32(std::int64_t v, std::int32_t n) {
  std::uint64_t uv = *reinterpret_cast<std::uint64_t*>(&v);
  std::uint32_t sign = cuda::std::signbit(v);
  std::uint64_t constexpr kValueMask = ~(std::uint64_t{1} << 63);
  // Remove the sign bit
  uv = uv & kValueMask;

  std::int32_t tail = std::max(n - kF32MantissaBits, 0);

  std::int64_t v0 = uv >> tail;
  std::uint32_t low = static_cast<uint32_t>(v0 & ~std::uint32_t{0});
  // Bring back the sign bit
  low |= (sign << 31);

  return cuda::std::bit_cast<std::int32_t>(low);
}

XGBOOST_HOST_DEV_INLINE std::int64_t RestoreFixed64(std::int32_t v, std::int32_t n) {
  std::uint64_t uv = *reinterpret_cast<std::uint32_t*>(&v);
  std::uint32_t constexpr kValueMask = ~(std::uint32_t{1} << 31);
  // Remove the sign bit
  uv = uv & kValueMask;

  std::int32_t tail = std::max(n - kF32MantissaBits, 0);
  uv <<= tail;
  // Bring back the sign bit
  uv |= (std::uint64_t{cuda::std::signbit(v)} << 63);
  return uv;
}
}  // namespace detail

class GradientQuantiser {
 private:
  /* Convert gradient to fixed point representation. */
  GradientPairPrecise to_fixed_point_;
  /* Convert fixed point representation back to floating point. */
  GradientPairPrecise to_floating_point_;

 public:
  // Used for test
  GradientQuantiser(GradientPairPrecise to_fixed, GradientPairPrecise to_float)
      : to_fixed_point_{to_fixed}, to_floating_point_{to_float} {}
  GradientQuantiser(Context const* ctx, linalg::VectorView<GradientPair const> gpair,
                    MetaInfo const& info);
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64 ToFixedPoint(GradientPair const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairInt64
  ToFixedPoint(GradientPairPrecise const& gpair) const {
    auto adjusted = GradientPairInt64(gpair.GetGrad() * to_fixed_point_.GetGrad(),
                                      gpair.GetHess() * to_fixed_point_.GetHess());
    return adjusted;
  }
  [[nodiscard]] XGBOOST_DEVICE GradientPairPrecise
  ToFloatingPoint(const GradientPairInt64& gpair) const {
    auto g = gpair.GetQuantisedGrad() * to_floating_point_.GetGrad();
    auto h = gpair.GetQuantisedHess() * to_floating_point_.GetHess();
    return {g, h};
  }
};

// For vector leaf
class MultiGradientQuantiser {
 private:
  dh::device_vector<GradientQuantiser> quantizers_;

 public:
  MultiGradientQuantiser(Context const* ctx, linalg::MatrixView<GradientPair const> gpair,
                         MetaInfo const& info);

  [[nodiscard]] auto Quantizers() const { return dh::ToSpan(this->quantizers_); }
};

void CalcQuantizedGpairs(Context const* ctx, linalg::Matrix<GradientPair>* const gpairs,
                         common::Span<GradientQuantiser const> roundings,
                         linalg::Matrix<GradientPairInt64>* p_out);
}  // namespace xgboost::tree
