/**
 * Copyright 2026, XGBoost Contributors
 *
 * @brief Thin boosting-loop driver for fused K-fold cross-validation on the GPU `hist`
 *        method with external memory.
 *
 * `TrainFusedCV` owns the per-fold intercept estimation, per-fold gradient computation,
 * the fused tree update (`GPUFusedCVHistMaker`, one shared page pass per level for all
 * folds), the fused validation prediction (one shared pass per round for all folds), and
 * the per-fold validation metric. It deliberately does **not** touch the existing
 * `Learner` / `GBTree` flow — it is an additive POC entry point.
 *
 * POC scope: single shared `ExtMemQuantileDMatrix`, contiguous-block folds, scalar leaf,
 * `subsample = colsample_* = 1.0`. The validation metric is the RMSE of the raw margin
 * (the default for `reg:squarederror`); other metrics are future work.
 */
#ifndef XGBOOST_TREE_FUSED_CV_TRAINER_H_
#define XGBOOST_TREE_FUSED_CV_TRAINER_H_

#include <cstdint>  // for int32_t
#include <string>   // for string
#include <vector>   // for vector

#include "cv_fold_info.h"     // for CVFoldInfo
#include "xgboost/base.h"     // for Args
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for DMatrix

namespace xgboost::tree {
/**
 * @brief Cross-validation metric history, shaped like `xgboost.cv`'s output.
 */
struct CVResults {
  /** @brief Name of the evaluation metric. */
  std::string metric;
  std::int32_t num_boost_round{0};
  std::int32_t n_folds{0};
  /** @brief Per-iteration mean of the validation metric across folds (size num_round). */
  std::vector<double> test_mean;
  /** @brief Per-iteration (population) standard deviation across folds. */
  std::vector<double> test_std;
  /** @brief Raw per-fold validation metric, indexed `[iteration][fold]`. */
  std::vector<std::vector<double>> per_fold;
};

/**
 * @brief Run fused K-fold cross-validation training and return the per-iteration metric
 *        history.
 *
 * @param ctx             CUDA context.
 * @param p_fmat          The single shared matrix over all train + validation rows.
 * @param folds           Contiguous-block fold layout over `p_fmat`'s rows.
 * @param params          Training parameters (objective + tree params), as key/value pairs.
 * @param num_boost_round Number of boosting rounds.
 * @param metric          Evaluation metric name; empty selects the objective's default.
 */
CVResults TrainFusedCV(Context const* ctx, DMatrix* p_fmat, CVFoldInfo const& folds,
                       Args const& params, std::int32_t num_boost_round,
                       std::string const& metric = "");
}  // namespace xgboost::tree

#endif  // XGBOOST_TREE_FUSED_CV_TRAINER_H_
