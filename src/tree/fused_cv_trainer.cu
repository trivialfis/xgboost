/**
 * Copyright 2026, XGBoost Contributors
 */
#include "fused_cv_trainer.h"

#include <algorithm>  // for max
#include <cmath>      // for sqrt
#include <memory>     // for unique_ptr, make_unique
#include <numeric>    // for partial_sum
#include <string>     // for string
#include <utility>    // for pair
#include <vector>     // for vector

#include "../common/cuda_rt_utils.h"     // for SetDevice
#include "../data/batch_utils.h"         // for StaticBatch
#include "../data/ellpack_page.cuh"      // for EllpackPageImpl
#include "../data/ellpack_page.h"        // for EllpackPage
#include "hist/hist_param.h"             // for HistMakerTrainParam
#include "param.h"                       // for TrainParam
#include "updater_gpu_hist_cv.cuh"       // for GPUFusedCVHistMaker, PredictTreeBinned
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/linalg.h"              // for Matrix, Vector, MakeTensorView
#include "xgboost/objective.h"           // for ObjFunction
#include "xgboost/tree_model.h"          // for RegTree

namespace xgboost::tree {
namespace {
using xgboost::cuda_impl::StaticBatch;

std::string FindParam(Args const& params, std::string const& key, std::string const& dft) {
  for (auto const& kv : params) {
    if (kv.first == key) {
      return kv.second;
    }
  }
  return dft;
}

// Shared cuts + per-batch global prefix-sum + dense flag, mirroring `InitBatchCuts` in the
// production updater (the matrix already holds the cuts; we only collect them once).
struct SharedSetup {
  std::shared_ptr<common::HistogramCuts const> cuts;
  std::vector<bst_idx_t> batch_ptr{0};
  bool dense{false};
};

SharedSetup CollectSharedSetup(Context const* ctx, DMatrix* p_fmat) {
  SharedSetup s;
  std::int32_t dense_compressed = -1;
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(false))) {
    s.batch_ptr.push_back(page.Size());
    s.cuts = page.Impl()->CutsShared();
    dense_compressed = page.Impl()->IsDenseCompressed();
  }
  std::partial_sum(s.batch_ptr.cbegin(), s.batch_ptr.cend(), s.batch_ptr.begin());
  s.dense = static_cast<bool>(dense_compressed);
  return s;
}

// Estimate fold `f`'s intercept (in margin space) from that fold's training labels, matching
// `Learner::FitIntercept` (`InitEstimation` then `ProbToMargin`).
float FoldIntercept(Context const* ctx, MetaInfo const& info, ObjFunction* obj,
                    CVFoldInfo const& folds, std::int32_t f) {
  std::vector<bst_idx_t> train_ridx;
  train_ridx.reserve(folds.TrainRows(f));
  for (auto const& r : folds.TrainRanges(f)) {
    for (bst_idx_t i = r.first; i < r.second; ++i) {
      train_ridx.push_back(i);
    }
  }
  MetaInfo fold_info = info.Slice(ctx, common::Span<bst_idx_t const>{train_ridx}, 0);
  linalg::Vector<float> base_score;
  obj->InitEstimation(fold_info, &base_score);
  obj->ProbToMargin(&base_score);
  return base_score.HostView()(0);
}

// Set `span[begin, begin + len)` to a zero gradient on the device.
void ZeroGpairRange(Context const* ctx, common::Span<GradientPair> span, bst_idx_t begin,
                    bst_idx_t len) {
  if (len == 0) {
    return;
  }
  dh::LaunchN(len, ctx->CUDACtx()->Stream(),
              [=] __device__(std::size_t i) { span[begin + i] = GradientPair{0.0f, 0.0f}; });
}

// RMSE of the raw margin over fold `f`'s validation rows.
double EvalFoldRmse(HostDeviceVector<float> const& margin, std::vector<float> const& labels,
                    CVFoldInfo const& folds, std::int32_t f) {
  auto const& h_margin = margin.ConstHostVector();
  double sum_sq = 0.0;
  bst_idx_t begin = folds.valid_ptr[f];
  bst_idx_t end = folds.valid_ptr[f + 1];
  for (bst_idx_t i = begin; i < end; ++i) {
    double d = static_cast<double>(h_margin[i]) - static_cast<double>(labels[i]);
    sum_sq += d * d;
  }
  bst_idx_t n = end - begin;
  return n == 0 ? 0.0 : std::sqrt(sum_sq / static_cast<double>(n));
}
}  // anonymous namespace

CVResults TrainFusedCV(Context const* ctx, DMatrix* p_fmat, CVFoldInfo const& folds,
                       Args const& params, std::int32_t num_boost_round,
                       std::string const& metric) {
  CHECK(ctx->IsCUDA()) << "Fused CV is implemented for the GPU `hist` method only.";
  curt::SetDevice(ctx->Ordinal());
  std::int32_t const K = folds.n_folds;
  CHECK_GE(K, 1);
  auto& info = p_fmat->Info();
  CHECK_EQ(info.num_row_, folds.n_rows) << "Fold layout does not match the matrix row count.";
  info.feature_types.SetDevice(ctx->Device());

  // ---- Objective, parameters, metric ----
  std::string obj_name = FindParam(params, "objective", "reg:squarederror");
  std::unique_ptr<ObjFunction> obj{ObjFunction::Create(obj_name, ctx)};
  obj->Configure(params);
  std::string metric_name = metric.empty() ? std::string{obj->DefaultEvalMetric()} : metric;
  CHECK_EQ(metric_name, "rmse")
      << "The fused CV POC supports the `rmse` metric only; got `" << metric_name << "`.";

  TrainParam param;
  param.UpdateAllowUnknown(params);
  HistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(params);

  auto setup = CollectSharedSetup(ctx, p_fmat);
  auto n_features = static_cast<bst_feature_t>(info.num_col_);
  GPUFusedCVHistMaker maker{ctx,       param,       &hist_param, folds,
                            setup.batch_ptr, setup.cuts, setup.dense, n_features};

  // Host copy of labels for the (single-target) RMSE evaluation.
  std::vector<float> h_labels = info.labels.Data()->ConstHostVector();

  // ---- Per-fold state ----
  std::vector<float> intercept(K);
  std::vector<HostDeviceVector<float>> train_margin(K);
  std::vector<HostDeviceVector<float>> valid_margin(K);
  std::vector<linalg::Matrix<GradientPair>> gpair(K);
  std::vector<HostDeviceVector<bst_node_t>> position(K);
  std::vector<std::vector<RegTree>> forest(K);
  for (std::int32_t f = 0; f < K; ++f) {
    intercept[f] = FoldIntercept(ctx, info, obj.get(), folds, f);
    train_margin[f].SetDevice(ctx->Device());
    train_margin[f].Resize(folds.n_rows, intercept[f]);
    valid_margin[f].SetDevice(ctx->Device());
    valid_margin[f].Resize(folds.n_rows, intercept[f]);
    gpair[f] = linalg::Matrix<GradientPair>{{folds.n_rows, bst_idx_t{1}}, ctx->Device()};
    forest[f].reserve(num_boost_round);
  }

  CVResults results;
  results.metric = metric_name;
  results.num_boost_round = num_boost_round;
  results.n_folds = K;
  results.per_fold.assign(num_boost_round, std::vector<double>(K, 0.0));
  results.test_mean.assign(num_boost_round, 0.0);
  results.test_std.assign(num_boost_round, 0.0);

  // ---- Boosting loop ----
  for (std::int32_t t = 0; t < num_boost_round; ++t) {
    std::vector<HostDeviceVector<GradientPair>*> gptr;
    std::vector<RegTree*> tptr;
    std::vector<HostDeviceVector<bst_node_t>*> pptr;
    std::vector<RegTree const*> new_trees;
    for (std::int32_t f = 0; f < K; ++f) {
      // 1. Per-fold gradient over the global margin, then zero the validation rows so they
      //    are not in any partitioner and do not perturb the (integer) histogram sums.
      obj->GetGradient(train_margin[f], info, t, &gpair[f]);
      ZeroGpairRange(ctx, gpair[f].Data()->DeviceSpan(), folds.valid_ptr[f],
                     folds.valid_ptr[f + 1] - folds.valid_ptr[f]);
      forest[f].emplace_back();
      gptr.push_back(gpair[f].Data());
      tptr.push_back(&forest[f].back());
      pptr.push_back(&position[f]);
      new_trees.push_back(&forest[f].back());
    }

    // 2. Fused tree update: one shared page pass per level grows all K trees.
    maker.UpdateTrees(p_fmat, gptr, tptr, pptr);

    // 3. Incrementally update each fold's training margin (train rows only).
    for (std::int32_t f = 0; f < K; ++f) {
      auto view = linalg::MakeTensorView(ctx, &train_margin[f], folds.n_rows, bst_target_t{1});
      maker.UpdatePredictionCache(f, view, &forest[f].back());
    }

    // 4. Fused validation prediction: one shared page pass updates every fold's margin.
    std::vector<HostDeviceVector<float>*> vptr;
    vptr.reserve(K);
    for (std::int32_t f = 0; f < K; ++f) {
      vptr.push_back(&valid_margin[f]);
    }
    maker.PredictValidationBinned(p_fmat, new_trees, vptr);

    // 5. Per-fold metric + aggregation.
    double mean = 0.0;
    for (std::int32_t f = 0; f < K; ++f) {
      double m = EvalFoldRmse(valid_margin[f], h_labels, folds, f);
      results.per_fold[t][f] = m;
      mean += m;
    }
    mean /= static_cast<double>(K);
    double var = 0.0;
    for (std::int32_t f = 0; f < K; ++f) {
      double d = results.per_fold[t][f] - mean;
      var += d * d;
    }
    var /= static_cast<double>(K);
    results.test_mean[t] = mean;
    results.test_std[t] = std::sqrt(var);
  }

  return results;
}
}  // namespace xgboost::tree
