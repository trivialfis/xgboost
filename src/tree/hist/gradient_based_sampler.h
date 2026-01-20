/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_GRADIENT_BASED_SAMPLER_H_
#define XGBOOST_TREE_HIST_GRADIENT_BASED_SAMPLER_H_

#include "xgboost/base.h"     // for GradientPair
#include "xgboost/context.h"  // for Context
#include "xgboost/linalg.h"   // for MatrixView

namespace xgboost::tree::cpu_impl {
/**
 * @brief Default lambda for MVS regularization when adaptive lambda is not available.
 *
 * This value (0.1) is the default used in CatBoost and shows good performance on most datasets.
 */
constexpr float kDefaultMvsLambda = 0.1f;

/**
 * @brief Sample gradients using gradient-based (MVS) sampling.
 *
 * Samples rows with probability proportional to their regularized absolute gradient.
 * Selected rows have their gradients scaled by 1/p to maintain unbiased estimates.
 * For multi-target, the importance is summed across targets.
 *
 * @param ctx Execution context for parallelism
 * @param gpairs Gradient pairs to sample (modified in place)
 * @param subsample Subsample ratio (0, 1]
 * @param mvs_lambda Lambda parameter for MVS regularization. Controls the tradeoff between
 *                   gradient importance and sample size distribution. If <= 0, uses default 0.1.
 */
void GradientBasedSample(Context const* ctx, linalg::MatrixView<GradientPair> gpairs,
                         float subsample, float mvs_lambda);
}  // namespace xgboost::tree::cpu_impl

#endif  // XGBOOST_TREE_HIST_GRADIENT_BASED_SAMPLER_H_
