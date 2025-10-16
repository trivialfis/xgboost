/**
 * Copyright 2025, XGBoost Contributors
 */
#include "../common/device_vector.cuh"
#include "../tree/tree_view.h"
#include "tree_cache.h"

namespace xgboost::gbm::cuda_impl {
using TreeViewVar = ::xgboost::gbm::TreeCache::DeviceTreeViewVar;

struct TreeCache {
  dh::device_vector<TreeViewVar> d_trees;

  std::size_t Set(Context const* ctx, std::vector<std::unique_ptr<RegTree>> const& p_trees,
                  bst_tree_t tree_begin, bst_tree_t tree_end) {
    std::vector<TreeViewVar> h_trees;
    std::size_t n_nodes = 0;
    for (bst_tree_t tree_idx = tree_begin; tree_idx < tree_end; ++tree_idx) {
      auto const& p_tree = p_trees[tree_idx];
      if (p_tree->IsMultiTarget()) {
        auto d_tree = tree::MultiTargetTreeView{ctx, p_tree.get()};
        n_nodes += d_tree.Size();
        h_trees.emplace_back(d_tree);
      } else {
        auto d_tree = tree::ScalarTreeView{ctx, p_tree.get()};
        n_nodes += d_tree.Size();
        h_trees.emplace_back(d_tree);
      }
    }

    this->d_trees = h_trees;
    return n_nodes;
  }
};
}  // namespace xgboost::gbm::cuda_impl
