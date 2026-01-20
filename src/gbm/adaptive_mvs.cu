/**
 * Copyright 2026, XGBoost Contributors
 */
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include <cub/cub.cuh>
#include <cuda/std/cmath>  // for sqrt

#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../tree/tree_view.h"         // for ScalarTreeView, MultiTargetTreeView
#include "xgboost/base.h"              // for GradientPair
#include "xgboost/context.h"           // for Context
#include "xgboost/linalg.h"            // for MatrixView
#include "xgboost/tree_model.h"        // for RegTree

namespace xgboost::gbm::cuda_impl {
/**
 * @brief Compute mean of sqrt(sum_g2 / sum_h2) over all rows.
 *
 * For each row, sums g^2 and h^2 across all targets, then computes sqrt(sum_g2 / sum_h2).
 * Returns the mean of this value across all rows.
 */
[[nodiscard]] double MeanGradSqrt(Context const* ctx,
                                  linalg::MatrixView<GradientPair const> gpairs) {
  auto n_samples = gpairs.Shape(0);
  auto n_targets = gpairs.Shape(1);
  auto cuctx = ctx->CUDACtx();

  // For each row, compute sqrt(sum_g2 / sum_h2)
  auto row_value_iter = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(std::size_t ridx) -> double {
        double sum_g2 = 0.0;
        double sum_h2 = 0.0;
        for (bst_target_t t = 0; t < n_targets; ++t) {
          auto gpair = gpairs(ridx, t);
          auto g = static_cast<double>(gpair.GetGrad());
          auto h = static_cast<double>(gpair.GetHess());
          sum_g2 += g * g;
          sum_h2 += h * h;
        }
        // Avoid division by zero
        if (sum_h2 == 0.0) {
          return 0.0;
        }
        return cuda::std::sqrt(sum_g2 / sum_h2);
      });

  // Sum all row values
  double sum = thrust::reduce(cuctx->CTP(), row_value_iter, row_value_iter + n_samples, 0.0,
                              cuda::std::plus{});

  return sum / static_cast<double>(n_samples);
}

/**
 * @brief Compute mean of leaf values for adaptive MVS lambda on GPU.
 *
 * For scalar trees: returns mean of leaf values
 * For multi-target trees: returns mean of sqrt(sum of squared leaf values across targets)
 */
[[nodiscard]] double MeanLeafSqrt(Context const* ctx, RegTree const& tree) {
  auto cuctx = ctx->CUDACtx();
  auto n_nodes = tree.NumNodes();
  auto n_leaves = tree.GetNumLeaves();

  if (tree.IsMultiTarget()) {
    // Multi-target tree
    tree::MultiTargetTreeView view{ctx->Device(), false, &tree};
    auto n_targets = view.NumTargets();
    // For each node, compute sqrt(sum of squared leaf values) if it's a leaf, else 0
    auto node_value_iter = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(bst_node_t nidx) -> double {
          if (!view.IsLeaf(nidx)) {
            return 0.0;
          }
          auto w = view.LeafValue(nidx);
          double sum_sq = 0.0;
          for (bst_target_t t = 0; t < n_targets; ++t) {
            auto v = static_cast<double>(w(t));
            sum_sq += v * v;
          }
          return cuda::std::sqrt(sum_sq);
        });
    double sum = thrust::reduce(cuctx->CTP(), node_value_iter, node_value_iter + n_nodes, 0.0,
                                cuda::std::plus{});
    return sum / static_cast<double>(n_leaves);
  } else {
    // Scalar tree
    tree::ScalarTreeView view{ctx->Device(), false, &tree};
    // For each node, return leaf value if it's a leaf, else 0
    auto node_value_iter = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(bst_node_t nidx) -> double {
          if (!view.IsLeaf(nidx)) {
            return 0.0;
          }
          return static_cast<double>(view.LeafValue(nidx));
        });
    double sum = thrust::reduce(cuctx->CTP(), node_value_iter, node_value_iter + n_nodes, 0.0,
                                cuda::std::plus{});
    return sum / static_cast<double>(n_leaves);
  }
}
}  // namespace xgboost::gbm::cuda_impl
