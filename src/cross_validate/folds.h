/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once
#include "xgboost/base.h"

namespace xgboost::cv {
struct Segment {
  bst_idx_t beg;
  bst_idx_t cnt;

  Segment(bst_idx_t beg, bst_idx_t cnt) : beg{beg}, cnt{cnt} {}
  [[nodiscard]] auto End() const { return beg + cnt; }
};
}  // namespace xgboost::cv
