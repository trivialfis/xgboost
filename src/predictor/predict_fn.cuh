#pragma once
#include <cstdint>
#include <cuda/pipeline>
#include <cuda/std/variant>

#include "../tree/tree_view.h"
#include "xgboost/span.h"

namespace xgboost::predictor {
// An heuristic for how many nodes to load into shared memory. Tunable.
template <std::uint32_t kBlockThreads>
std::size_t constexpr TreeShmemNodes() {
  return kBlockThreads;
}

template <std::uint32_t kBlockThreads, typename TreeViewVar, typename Fn>
__device__ void ForEachTree(common::Span<TreeViewVar const> d_trees, RegTree::Node* smem, Fn&& fn) {
  if (d_trees.size() <= 2 || !smem) {
    for (bst_tree_t tree_idx = 0; tree_idx < d_trees.size(); ++tree_idx) {
      auto tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]);
      fn(tree, tree_idx);
    }
    return;
  }

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

  constexpr auto kNodesMax = TreeShmemNodes<kBlockThreads>();
  constexpr auto kNodeSize = sizeof(RegTree::Node);

  auto load = [&pipe, &smem](tree::ScalarTreeView const& tree, bst_tree_t tree_idx) {
    auto stage = tree_idx % 2 == 0;
    auto dst = smem + stage * kNodesMax + threadIdx.x;
    if (threadIdx.x < std::min(tree.n, static_cast<bst_node_t>(kNodesMax))) {
      cuda::memcpy_async(dst, &tree.nodes[threadIdx.x], kNodeSize, pipe);
    }
  };
  auto create_tree_view = [&](bst_tree_t tree_idx) {
    auto stage = tree_idx % 2 == 0;
    auto dst = smem + stage * kNodesMax;
    auto tree = cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]);
    return tree::ScalarTreeView{dst, tree.stats, tree.GetCategoriesMatrix(),
                                std::min(tree.n, static_cast<bst_node_t>(kNodesMax))};
  };

  bst_tree_t tree_idx = 0;

  pipe.producer_acquire();
  load(cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]), tree_idx);
  pipe.producer_commit();
  tree_idx += 1;

  pipe.producer_acquire();
  load(cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx]), tree_idx);
  pipe.producer_commit();
  tree_idx += 1;

  for (bst_tree_t tree_idx = 0; tree_idx < d_trees.size(); ++tree_idx) {
    cuda::pipeline_consumer_wait_prior<1>(pipe);
    __syncthreads();

    auto tree = create_tree_view(tree_idx);
    fn(tree, tree_idx);
    pipe.consumer_release();
    __syncthreads();

    pipe.producer_acquire();
    if (tree_idx + 2 < d_trees.size()) {
      load(cuda::std::get<tree::ScalarTreeView>(d_trees[tree_idx + 2]), tree_idx + 2);
    }
    pipe.producer_commit();
  }
}
}  // namespace xgboost::predictor
