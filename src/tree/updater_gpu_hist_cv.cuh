/**
 * Copyright 2026, XGBoost Contributors
 *
 * @brief Self-contained fused cross-validation tree maker for the GPU `hist` method.
 *
 * Grows K trees (one per CV fold) per boosting round from a single shared
 * `ExtMemQuantileDMatrix`, reusing each fetched Ellpack page across all folds (one fetch per
 * tree level for all folds combined). This is the realization of the "fuse CV by reusing
 * quantiles + pages" optimization.
 *
 * It does **not** modify or reuse `GPUHistMakerDevice` (which is translation-unit-local).
 * Instead it reuses the shared leaf-level building blocks (`DeviceHistogramBuilder`,
 * `GPUHistEvaluator`, `RowPartitionerBatches`, `GradientQuantiserGroup`, `Driver`,
 * `FeatureGroups`, and the header helpers `EncodeOp` / `GoLeftWrapperOp` / `AssignNodes`)
 * directly, per fold, and orchestrates the level-synchronized shared-page loop itself.
 */
#ifndef XGBOOST_TREE_UPDATER_GPU_HIST_CV_CUH_
#define XGBOOST_TREE_UPDATER_GPU_HIST_CV_CUH_

#include <memory>  // for unique_ptr, shared_ptr
#include <vector>  // for vector

#include "cv_fold_info.h"                // for CVFoldInfo, FoldBatchView
#include "gpu_hist/evaluate_splits.cuh"  // for GPUHistEvaluator
#include "gpu_hist/expand_entry.cuh"     // for GPUExpandEntry
#include "gpu_hist/quantiser.cuh"        // for GradientQuantiserGroup
#include "param.h"                       // for TrainParam
#include "updater_gpu_hist.cuh"          // for EncodeOp, AssignNodes, kMaxNodeBatchSize, ...
#include "xgboost/base.h"                // for bst_idx_t, GradientPairInt64
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for DMatrix, MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix, MatrixView
#include "xgboost/tree_model.h"          // for RegTree

namespace xgboost::tree {
/**
 * @brief Per-fold device state. Each fold owns its own building blocks so nothing mutable is
 *        shared across the sequentially-processed folds (the read-only cuts and feature
 *        groups are shared by the maker).
 */
struct FoldDeviceState {
  GPUHistEvaluator evaluator;
  RowPartitionerBatches partitioners;  // seeded with the fold's TRAIN rows per source batch
  DeviceHistogramBuilder histogram;
  std::unique_ptr<GradientQuantiserGroup> quantiser;
  std::shared_ptr<common::ColumnSampler> column_sampler;  // per fold (determinism)
  FeatureInteractionConstraintDevice interaction_constraints;
  Driver<GPUExpandEntry> driver;

  // Global-sized (N) quantised gradient; only the fold's training rows are populated, the
  // rest are zero (and never referenced because they are not in any partitioner).
  linalg::Matrix<GradientPairInt64> d_gpair;
  // Per-batch training-row view of the fold over the shared pages.
  FoldBatchView view;

  // Per-level bookkeeping (reused across the level loop).
  std::vector<GPUExpandEntry> expand_set;
  std::vector<GPUExpandEntry> valid_candidates;
  std::vector<bst_node_t> build_nidx;
  std::vector<bst_node_t> subtraction_nidx;

  // Global-sized (N) leaf position for the fold's training rows (validation rows untouched).
  HostDeviceVector<bst_node_t>* p_out_position{nullptr};
  RegTree* tree{nullptr};
  bst_idx_t fold_rows{0};

  FoldDeviceState(Context const* ctx, TrainParam const& param, bst_feature_t n_features)
      : evaluator{param, n_features, ctx->Device()},
        column_sampler{std::make_shared<common::ColumnSampler>()},
        interaction_constraints(param, static_cast<std::int32_t>(n_features)),
        driver{param, cuda_impl::kMaxNodeBatchSize} {}

  [[nodiscard]] bool BatchActive(std::int32_t k) const { return view.per_batch_rows[k] > 0; }
};

class GPUFusedCVHistMaker {
  Context const* ctx_;
  TrainParam param_;
  HistMakerTrainParam const* hist_param_;
  CVFoldInfo folds_;
  std::vector<bst_idx_t> batch_ptr_;  // full-matrix per-batch global prefix sum
  std::shared_ptr<common::HistogramCuts const> cuts_;
  bool dense_compressed_;
  bst_feature_t n_features_;
  std::unique_ptr<FeatureGroups> feature_groups_;  // shared (depends only on cuts)
  std::vector<std::unique_ptr<FoldDeviceState>> fold_;
  dh::PinnedMemory pinned_;
  dh::PinnedMemory pinned2_;

  // ---- Per-fold leaf-level helpers (mirror GPUHistMakerDevice, single fold) ----
  void ResetFold(FoldDeviceState* st, MetaInfo const& info,
                 HostDeviceVector<GradientPair> const* gpair);
  [[nodiscard]] GradientPairInt64 RootSum(FoldDeviceState* st, MetaInfo const& info) const;
  void BuildHist(FoldDeviceState* st, EllpackPage const& page, std::int32_t k, bst_node_t nidx);
  [[nodiscard]] GPUExpandEntry EvaluateRootSplit(FoldDeviceState* st, DMatrix const* p_fmat,
                                                 GradientPairInt64 root_sum);
  void EvaluateSplits(FoldDeviceState* st, DMatrix const* p_fmat,
                      std::vector<GPUExpandEntry> const& candidates,
                      common::Span<GPUExpandEntry> out);
  void ApplySplit(FoldDeviceState* st, GPUExpandEntry const& candidate);
  void ReduceHist(FoldDeviceState* st, DMatrix* p_fmat, MetaInfo const& info);
  void FinalisePosition(FoldDeviceState* st);

 public:
  GPUFusedCVHistMaker(Context const* ctx, TrainParam param, HistMakerTrainParam const* hist_param,
                      CVFoldInfo folds, std::vector<bst_idx_t> batch_ptr,
                      std::shared_ptr<common::HistogramCuts const> cuts, bool dense_compressed,
                      bst_feature_t n_features);

  /**
   * @brief Grow one tree per fold from the shared matrix in a single level-synchronized,
   *        one-fetch-per-level page loop.
   *
   * @param gpair     Per-fold GLOBAL-sized (N rows) gradient, validation rows zeroed.
   * @param trees     Per-fold output tree.
   * @param positions Per-fold GLOBAL-sized leaf positions (only training rows written).
   */
  void UpdateTrees(DMatrix* p_fmat, std::vector<HostDeviceVector<GradientPair>*> const& gpair,
                   std::vector<RegTree*> const& trees,
                   std::vector<HostDeviceVector<bst_node_t>*> const& positions);

  /**
   * @brief Add fold `f`'s just-grown tree contribution to its training-margin cache.
   *
   * Driven over the fold's training rows only (re-implemented prediction-cache update,
   * review #2 R2-B) so the unwritten validation slots of the global position buffer are
   * never dereferenced.
   */
  void UpdatePredictionCache(std::int32_t f, linalg::MatrixView<float> out_preds,
                             RegTree const* p_tree);

  /**
   * @brief Fused validation prediction: one shared page pass adds every fold's just-grown
   *        tree to its own validation-margin cache.
   *
   * Each global row belongs to exactly one fold's validation block, so a single pass over
   * the shared pages predicts all K folds' newest trees on their respective validation rows
   * (success criterion #4 — validation prediction stays at one page pass per round for all
   * folds, never K passes). Traversal uses the binned Ellpack value (`GetFvalue` over cut
   * values), consistent with how the trees were grown.
   *
   * @param new_trees     Per-fold tree grown this round (size = NumFolds()).
   * @param valid_margins Per-fold GLOBAL-sized (N) validation margin; only the fold's
   *                      validation-row slots are updated.
   */
  void PredictValidationBinned(DMatrix* p_fmat, std::vector<RegTree const*> const& new_trees,
                               std::vector<HostDeviceVector<float>*> const& valid_margins);

  [[nodiscard]] CVFoldInfo const& Folds() const { return folds_; }
  [[nodiscard]] std::int32_t NumFolds() const { return folds_.n_folds; }
};

/**
 * @brief Add a single tree's contribution (binned traversal) to `out_margin` over **all**
 *        rows of `p_fmat`, indexed by the matrix's local row index.
 *
 * Shares the binned traversal with the fused validation predictor so a reference path can
 * predict bit-identical trees the same way (review #2 R2-D). `out_margin` must be sized to
 * `p_fmat->Info().num_row_` and live on `ctx`'s device.
 */
void PredictTreeBinned(Context const* ctx, DMatrix* p_fmat, RegTree const& tree,
                       common::Span<float> out_margin);
}  // namespace xgboost::tree

#endif  // XGBOOST_TREE_UPDATER_GPU_HIST_CV_CUH_
