/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once
#include <cassert>      // for assert
#include <cstdint>      // for int32_t
#include <type_traits>  // for remove_cv_t

#include "xgboost/linalg.h"  // for TensorView

namespace xgboost::linalg {
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
  XGBOOST_DEVICE auto Shape() const {
    if (k == 0) {
      return this->idx_.size();
    }
    return this->ten_.Shape(k);
  }
};
}  // namespace xgboost::linalg
