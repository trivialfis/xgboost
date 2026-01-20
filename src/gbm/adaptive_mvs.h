/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include "xgboost/base.h"        // for GradientPair
#include "xgboost/context.h"     // for Context
#include "xgboost/linalg.h"      // for MatrixView
#include "xgboost/tree_model.h"  // for RegTree

namespace xgboost::gbm {
namespace cpu_impl {
[[nodiscard]] double MeanGradSqrt(Context const* ctx,
                                  linalg::MatrixView<GradientPair const> gpairs);
[[nodiscard]] double MeanLeafSqrt(RegTree const& tree);
}  // namespace cpu_impl

#if defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
[[nodiscard]] double MeanGradSqrt(Context const* ctx,
                                  linalg::MatrixView<GradientPair const> gpairs);
[[nodiscard]] double MeanLeafSqrt(Context const* ctx, RegTree const& tree);
}  // namespace cuda_impl
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::gbm
