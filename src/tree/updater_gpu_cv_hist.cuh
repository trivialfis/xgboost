/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "../common/device_vector.cuh"         // for DeviceUVector
#include "../common/random.h"                  // for ColumnSampler
#include "../cross_validate/cross_validate.h"  // for FoldInfoBatches, FoldGpairs
#include "gpu_hist/evaluate_splits.cuh"        // for MultiEvaluateSplitSharedInputs
#include "gpu_hist/expand_entry.cuh"           // for MultiExpandEntry
#include "gpu_hist/feature_groups.cuh"         // for FeatureGroups
#include "gpu_hist/histogram.cuh"              // for DeviceHistogramBuilder
#include "gpu_hist/multi_evaluate_splits.cuh"  // for MultiHistEvaluator
#include "gpu_hist/quantiser.cuh"              // for GradientQuantiserGroup
#include "gpu_hist/row_partitioner.cuh"        // for RowPartitionerBatches, RowIndexT
#include "hist/hist_param.h"                   // for HistMakerTrainParam
#include "param.h"                             // for TrainParam
#include "xgboost/base.h"                      // for GradientPairInt64, bst_feature_t
#include "xgboost/context.h"                   // for Context
#include "xgboost/data.h"                      // for DMatrix
#include "xgboost/linalg.h"                    // for Matrix
#include "xgboost/span.h"                      // for Span
#include "xgboost/tree_model.h"                // for RegTree

namespace xgboost::tree::cuda_impl {
/**
 * @brief Calculate the root gradient sum for each fold.
 *
 * @param d_gpair  One quantised gradient matrix view per fold (global-sized, N x n_targets).
 * @param root_sum One output span per fold (size == n_targets).
 */
void CalcRootSumFolds(Context const* ctx,
                      std::vector<linalg::MatrixView<GradientPairInt64>> d_gpair,
                      std::vector<common::Span<GradientPairInt64>> root_sum);

/**
 * @brief Per-fold device state for the fused CV updater.
 *
 * Held via unique_ptr by the maker so the non-movable DeviceHistogramBuilder /
 * MultiHistEvaluator members never need to be moved.
 */
struct CvFoldDeviceState {
  // One RowPartitioner per source Ellpack batch, seeded with the fold's global training
  // row indices for that batch.
  RowPartitionerBatches partitioners;
  DeviceHistogramBuilder histogram;
  // Vector-leaf evaluator; handles n_targets >= 1, so single-target is just n_targets == 1.
  MultiHistEvaluator evaluator;
  // Quantiser built from the fold's (compact) gradient; one entry per target.
  std::unique_ptr<GradientQuantiserGroup> quantiser;
  // Global-sized (N x n_targets) quantised gradient; only the fold's training rows are non-zero.
  linalg::Matrix<GradientPairInt64> d_gpair;
  // Global training row indices, concatenated in source-batch order. Doubles as the
  // scatter map (build d_gpair) and the per-batch partitioner seed.
  dh::DeviceUVector<RowIndexT> grid;
  // Quantised root sum, one entry per target.
  dh::DeviceUVector<GradientPairInt64> root_sum;
};

/**
 * @brief Fused cross-validation GPU hist tree maker (root initialization).
 *
 * Grows the root of one tree per fold from a single shared ExtMemQuantileDMatrix. Each
 * source Ellpack page is fetched once and reused by every fold to build its root
 * histogram (the fusion), with no Ellpack data copy.
 *
 * Everything runs through the vector-leaf (multi-target) machinery; single-target is
 * simply the trivial case where n_targets == 1.
 */
class FusedCvHistTreeMaker {
  Context const* ctx_;
  TrainParam param_;
  HistMakerTrainParam const* hist_param_;
  std::shared_ptr<common::HistogramCuts const> cuts_;  // shared across folds
  std::unique_ptr<FeatureGroups> feature_groups_;      // shared (depends only on cuts)
  bool dense_compressed_{false};
  std::vector<bst_idx_t> batch_ptr_;  // source-batch prefix-sum (global row offsets)
  std::size_t k_folds_{0};
  bst_target_t n_targets_{1};                              // set in Reset from the fold gradients
  std::shared_ptr<common::ColumnSampler> column_sampler_;  // shared (colsample == 1 in POC)

  std::vector<std::unique_ptr<CvFoldDeviceState>> folds_;

  // Shared split-evaluation inputs for fold k (mirrors MultiTargetHistMaker::MakeSharedInputs).
  [[nodiscard]] MultiEvaluateSplitSharedInputs MakeSharedInputs(std::size_t k,
                                                                bst_feature_t max_active_feature);
  // Build the root histogram of every fold, fetching each source page once.
  void BuildRootHist(DMatrix* p_fmat);
  // Evaluate the root split for one fold and set its root leaf/stat.
  MultiExpandEntry EvaluateRoot(DMatrix const* p_fmat, std::size_t k, RegTree* p_tree);

 public:
  FusedCvHistTreeMaker(Context const* ctx, TrainParam param, HistMakerTrainParam const* hist_param,
                       std::shared_ptr<common::ColumnSampler> column_sampler,
                       std::vector<bst_idx_t> batch_ptr,
                       std::shared_ptr<common::HistogramCuts const> cuts, bool dense_compressed,
                       std::size_t k_folds);

  // Per-iteration setup: quantise gradients, build the global d_gpair, seed partitioners,
  // reset histograms/evaluators.
  void Reset(DMatrix* p_fmat, cv::FoldInfoBatches const& finfo, cv::FoldGpairs const& gpairs);

  // Root sum + root histogram + root split for every fold. Returns one candidate per fold.
  std::vector<MultiExpandEntry> InitRoots(DMatrix* p_fmat, std::vector<RegTree*> const& trees);

  // Accessors for testing.
  [[nodiscard]] std::size_t KFolds() const { return this->k_folds_; }
  [[nodiscard]] bst_target_t NTargets() const { return this->n_targets_; }
  [[nodiscard]] common::Span<GradientPairInt64 const> RootHistogram(std::size_t k);
  [[nodiscard]] RowPartitionerBatches& Partitioners(std::size_t k);
  // Dequantised root sum for target t of fold k.
  [[nodiscard]] GradientPairPrecise RootSum(std::size_t k, bst_target_t t = 0) const;
  // Per-target quantiser for fold k (used to dequantise the root histogram in tests).
  [[nodiscard]] GradientQuantiser const& Quantiser(std::size_t k, bst_target_t t = 0) const;
};
}  // namespace xgboost::tree::cuda_impl
