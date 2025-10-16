/**
 * Copyright 2025, XGBoost Contributors
 */
#pragma once
#include <cuda/std/variant>  // for variant
#include <variant>           // for variant

#include "../tree/tree_view.h"  // for ScalarTreeView, MultiTargetTreeView
#include "xgboost/span.h"       // for Span

namespace xgboost::gbm {
namespace cuda_impl {
struct TreeCache;
};

/**
 * @brief Device cache for tree views.
 *
 * This helps avoid repeated copies of tree view objects. The copy is not expensive, but
 * it blocks other copies and prevents pipelined prediction.
 */
struct TreeCache {
#if defined(XGBOOST_USE_CUDA)
  using DeviceTreeViewVar = cuda::std::variant<tree::ScalarTreeView, tree::MultiTargetTreeView>;
  common::Span<DeviceTreeViewVar const> d_trees;
#endif

  using HostTreeViewVar = std::variant<tree::ScalarTreeView, tree::MultiTargetTreeView>;
  common::Span<HostTreeViewVar const> h_trees;

  bst_tree_t beg{0};
  bst_tree_t end{0};

  [[nodiscard]] bool IsHot(bst_tree_t tree_begin, bst_tree_t tree_end) const {
    return tree_begin >= this->beg && tree_end <= this->end;
  }
};
}  // namespace xgboost::gbm
