/**
 * Copyright 2026, XGBoost Contributors
 */
#include <vector>  // for vector

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for caching_device_vector, LaunchN, ToSpan
#include "xgboost/context.h"             // for Context
#include "xgboost/tree_model.h"          // for RegTree, RTreeNodeStat

namespace xgboost::tree::cuda_impl {
namespace {
struct NodeStat {
  bst_node_t nidx;
  float const* weight;
  float loss_chg;
  float sum_hess;
};
}  // namespace

void SetNodeStats(Context const* ctx, ExpandBatch const& batch,
                  common::Span<RegTree::Node const> nodes, common::Span<RTreeNodeStat> stats) {
  std::vector<NodeStat> inputs;
  inputs.reserve(batch.size() * 3);
  for (auto const& entry : batch) {
    auto nidx = entry.split.nidx;
    inputs.push_back({nidx, entry.parent.weight.data(), entry.loss_chg,
                      static_cast<float>(entry.parent.sum_hess)});
    inputs.push_back({nodes[nidx].LeftChild(), entry.left.weight.data(), 0.0f,
                      static_cast<float>(entry.left.sum_hess)});
    inputs.push_back({nodes[nidx].RightChild(), entry.right.weight.data(), 0.0f,
                      static_cast<float>(entry.right.sum_hess)});
  }
  dh::caching_device_vector<NodeStat> storage(inputs);
  auto entries = dh::ToSpan(storage);
  dh::LaunchN(entries.size(), ctx->CUDACtx()->Stream(), [=] __device__(std::size_t i) {
    auto entry = entries[i];
    auto& stat = stats[entry.nidx];
    stat.loss_chg = entry.loss_chg;
    stat.sum_hess = entry.sum_hess;
    stat.base_weight = *entry.weight;
    stat.leaf_child_cnt = 0;
  });
}
}  // namespace xgboost::tree::cuda_impl
