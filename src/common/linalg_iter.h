/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once
#include <cassert>      // for assert
#include <cstdint>      // for int32_t
#include <type_traits>  // for remove_cv_t

#include "xgboost/linalg.h"  // for TensorView

namespace xgboost::linalg {
// Supports only permuting at the first dimension, also, read-only.
template <typename T, std::int32_t D>
class PermutationTensorView {
 public:
  using element_type = T;                  // NOLINT
  using value_type = std::remove_cv_t<T>;  // NOLINT

 private:
  linalg::TensorView<element_type, D> ten_;
  common::Span<bst_idx_t const> idx_;

 public:
  PermutationTensorView(linalg::TensorView<element_type, D> ten, common::Span<bst_idx_t const> idx)
      : ten_{std::move(ten)}, idx_{std::move(idx)} {
    assert(ten_.Shape(0) <= idx_.size());
  }

  template <std::int32_t k>
  [[nodiscard]]XGBOOST_DEVICE auto Shape() const {
    if (k == 0) {
      return this->idx_.size();
    }
    return this->ten_.Shape(k);
  }
  [[nodiscard]] XGBOOST_DEVICE auto Size() const {
    std::size_t shape[D];
    shape[0] = idx_.size();

    for (std::int32_t k = 1; k < D; ++k) {
      shape[k] = this->ten_.Shape(k);
    }
    auto size = detail::CalcSize(shape);
    return size;
  }
  template <typename Head, typename... Index>
  XGBOOST_DEVICE T const &operator()(Head &&head, Index &&...index) const {
    static_assert(sizeof...(index) + 1 <= D, "Invalid index.");
    auto idx = this->idx_[std::forward<Head>(head)];
    return this->ten_(idx, std::forward<Index>(index)...);
  }
};
}  // namespace xgboost::linalg
