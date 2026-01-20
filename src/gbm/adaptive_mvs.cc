/**
 * Copyright 2026, XGBoost Contributors
 */
#include <cmath>  // for sqrt
#include <numeric>

#include "../common/linalg_op.h"
#include "../encoder/types.h"
#include "../tree/tree_view.h"
#include "xgboost/base.h"        // for GradientPairPrecise
#include "xgboost/linalg.h"      // for MatrixView
#include "xgboost/tree_model.h"  // for Iter

namespace xgboost::gbm {
namespace cpu_impl {
[[nodiscard]] double MeanGradSqrt(Context const* ctx,
                                  linalg::MatrixView<GradientPair const> gpairs) {
  auto n_samples = gpairs.Shape(0);
  std::size_t constexpr kBlockOfRowsSize = 128;
  common::MemStackAllocator<double, common::DefaultMaxThreads()> tloc_sum(ctx->Threads(), 0.0);
  common::ParallelFor1d<kBlockOfRowsSize>(n_samples, ctx->Threads(), [&](auto&& block) {
    double blk_loc = 0;
    for (std::size_t i = block.begin(); i < block.end(); ++i) {
      // sum over all targets
      GradientPairPrecise t_sum;
      for (bst_target_t t = 0, n_targets = gpairs.Shape(1); t < n_targets; ++t) {
        auto v = gpairs(i, t);
        t_sum += GradientPairPrecise{v.GetGrad() * v.GetGrad(), v.GetHess() * v.GetHess()};
      }

      blk_loc += std::sqrt(t_sum.GetGrad() / t_sum.GetHess());
    }
    // Write to the thread local cache
    tloc_sum[omp_get_thread_num()] += blk_loc;
  });
  auto sum = std::accumulate(tloc_sum.cbegin(), tloc_sum.cend(), 0.0);
  return sum / n_samples;
}

[[nodiscard]] double MeanLeafSqrt(RegTree const& tree) {
  // fixme: need to consider random forest?
  double sum = 0;
  tree::WalkTree(tree, enc::Overloaded{[&](tree::ScalarTreeView tree, bst_node_t nidx) {
                                         if (tree.IsLeaf(nidx)) {
                                           auto w = tree.LeafValue(nidx);
                                           sum += w;
                                         }
                                         return true;
                                       },
                                       [&](tree::MultiTargetTreeView tree, bst_node_t nidx) {
                                         if (tree.IsLeaf(nidx)) {
                                           auto w = tree.LeafValue(nidx);
                                           double sum_leaf = 0;
                                           for (auto v : linalg::Iter(w)) {
                                             sum_leaf += v * v;
                                           }
                                           sum += std::sqrt(sum_leaf);
                                         }
                                         return true;
                                       }});
  return sum / tree.GetNumLeaves();
}
}  // namespace cpu_impl
}  // namespace xgboost::gbm
