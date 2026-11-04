/**
 * Copyright 2025-2026, XGBoost Contributors
 */
#include <ostream>  // for ostream

#include "expand_entry.cuh"

namespace xgboost::tree {
std::ostream& operator<<(std::ostream& os, GPUExpandEntry const& e) {
  return os << "GPUExpandEntry:\n"
            << "nidx: " << e.nidx << "\n"
            << "depth: " << e.depth << "\n"
            << "loss: " << e.split.loss_chg << "\n"
            << "left_hess: " << e.left_hess << "\n"
            << "right_hess: " << e.right_hess << "\n";
}
}  // namespace xgboost::tree
