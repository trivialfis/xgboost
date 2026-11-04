/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef EXPAND_ENTRY_CUH_
#define EXPAND_ENTRY_CUH_

#include <ostream>  // for ostream

#include "../updater_gpu_common.cuh"  // for DeviceSplitCandidate
#include "xgboost/base.h"             // for bst_node_t

namespace xgboost::tree {
/**
 * @brief Durable queue entry shared by scalar and vector GPU trees.
 *
 * Gradients, weights, and category bits are stored by node ID in the evaluator. Entries
 * must remain valid while waiting in the loss-guide queue, even when those buffers grow.
 */
struct GPUExpandEntry {
  bst_node_t nidx{0};
  bst_node_t depth{0};
  DeviceSplitCandidate split;

  // Sum of hessians across all split targets for left/right children.
  double left_hess{0};
  double right_hess{0};

  [[nodiscard]] float GetLossChange() const { return split.loss_chg; }
  [[nodiscard]] bst_node_t GetNodeId() const { return nidx; }

  friend std::ostream& operator<<(std::ostream& os, GPUExpandEntry const& entry);
};
}  // namespace xgboost::tree

#endif  // EXPAND_ENTRY_CUH_
