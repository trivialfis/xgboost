/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/gradient.h>      // for GradientContainer
#include <xgboost/task.h>          // for ObjInfo
#include <xgboost/tree_updater.h>  // for TreeUpdater

#include <memory>   // for shared_ptr
#include <numeric>  // for partial_sum
#include <string>   // for string, to_string
#include <vector>   // for vector

#include "../../../src/common/hist_util.h"               // for HistogramCuts
#include "../../../src/data/batch_utils.h"               // for StaticBatch
#include "../../../src/data/ellpack_page.cuh"            // for EllpackPageImpl
#include "../../../src/data/extmem_quantile_dmatrix.h"   // for ExtMemQuantileDMatrix
#include <xgboost/objective.h>  // for ObjFunction

#include "../../../src/tree/cv_fold_info.h"
#include "../../../src/tree/fused_cv_trainer.h"        // for TrainFusedCV, CVResults
#include "../../../src/tree/hist/hist_param.h"         // for HistMakerTrainParam
#include "../../../src/tree/param.h"                   // for TrainParam
#include "../../../src/tree/updater_gpu_hist_cv.cuh"   // for GPUFusedCVHistMaker, PredictTreeBinned
#include "../filesystem.h"                             // for TemporaryDirectory
#include "../helpers.h"
#include "fused_cv_test_helpers.h"

namespace xgboost::tree {
namespace {
// Collect the (shared) histogram cut values of an ExtMem QDM.
std::vector<float> GetCutValues(Context const* ctx, DMatrix* p_fmat, bst_bin_t bins) {
  auto param = BatchParam{bins, tree::TrainParam::DftSparseThreshold()};
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, param)) {
    return page.Cuts().cut_values_.ConstHostVector();
  }
  return {};
}

// Shared cuts + per-batch global prefix-sum + dense flag, mirroring `InitBatchCuts` in the
// production updater, using the same `StaticBatch` param the fused maker iterates with.
struct SharedSetup {
  std::shared_ptr<common::HistogramCuts const> cuts;
  std::vector<bst_idx_t> batch_ptr;
  bool dense{false};
};

SharedSetup MakeSharedSetup(Context const* ctx, DMatrix* full) {
  SharedSetup s;
  s.batch_ptr = {0};
  std::int32_t dense_compressed = -1;
  for (auto const& page : full->GetBatches<EllpackPage>(ctx, ::xgboost::cuda_impl::StaticBatch(false))) {
    s.batch_ptr.push_back(page.Size());
    s.cuts = page.Impl()->CutsShared();
    dense_compressed = page.Impl()->IsDenseCompressed();
  }
  std::partial_sum(s.batch_ptr.cbegin(), s.batch_ptr.cend(), s.batch_ptr.begin());
  s.dense = static_cast<bool>(dense_compressed);
  return s;
}

// Deterministic integer gradients (grad in {-1, +1}, hess = 1) so that the floating-point
// clipped-gradient sum the quantiser uses is exact regardless of row order — this guarantees
// the fused (global, zero-padded) and baseline (fold-local) quantisers are bit-identical.
std::vector<GradientPair> MakeGlobalGradients(cv_test::CVTestData const& data) {
  std::vector<GradientPair> g(data.n_rows);
  for (bst_idx_t i = 0; i < data.n_rows; ++i) {
    float grad = data.labels[i] >= 0.0f ? 1.0f : -1.0f;
    g[i] = GradientPair{grad, 1.0f};
  }
  return g;
}

GradientContainer MakeContainer(Context const* ctx, std::vector<GradientPair> const& v) {
  HostDeviceVector<GradientPair> tmp(v.size());
  tmp.HostVector() = v;
  GradientContainer gc;
  gc.gpair = linalg::Matrix<GradientPair>{{static_cast<bst_idx_t>(v.size()), bst_idx_t{1}},
                                          ctx->Device()};
  gc.gpair.Data()->Copy(tmp);
  return gc;
}

// Train one baseline tree via the public `grow_gpu_hist` updater over a standalone matrix.
RegTree TrainBaselineTree(Context const* ctx, TrainParam const& param,
                          std::vector<GradientPair> const& gpair, DMatrix* fold_fmat) {
  ObjInfo task{ObjInfo::kRegression};
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_gpu_hist", ctx, &task)};
  updater->Configure(Args{});
  auto gc = MakeContainer(ctx, gpair);
  RegTree tree;
  std::vector<HostDeviceVector<bst_node_t>> position(1);
  updater->Update(&param, &gc, fold_fmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                  {&tree});
  return tree;
}

TrainParam MakeTrainParam(bst_bin_t bins, std::int32_t max_depth) {
  Args args{{"max_depth", std::to_string(max_depth)},
            {"max_bin", std::to_string(bins)},
            {"min_child_weight", "0.0"},
            {"reg_alpha", "0"},
            {"reg_lambda", "0"},
            {"subsample", "1.0"},
            {"colsample_bynode", "1.0"},
            {"colsample_bylevel", "1.0"},
            {"colsample_bytree", "1.0"},
            {"seed", "1994"}};
  TrainParam param;
  param.UpdateAllowUnknown(args);
  return param;
}
}  // anonymous namespace

// Phase 0 smoke test: the helpers build a shared full matrix and a fold baseline that
// shares cuts, with the expected row counts.
TEST(GpuFusedCV, TestHelpersSmoke) {
  auto ctx = MakeCUDACtx(0);
  bst_idx_t constexpr kRows = 512;
  bst_feature_t constexpr kCols = 8;
  bst_bin_t constexpr kBins = 64;
  std::int32_t constexpr kFolds = 4;

  auto data = cv_test::MakeCVTestData(kRows, kCols);
  auto folds = CVFoldInfo::MakeContiguous(kRows, kFolds);

  common::TemporaryDirectory tmpdir;
  auto full = cv_test::MakeExtMemQdm(&ctx, data, /*n_batches=*/4, kBins, /*on_host=*/false,
                                     tmpdir.Str() + "/full");
  ASSERT_EQ(full->Info().num_row_, kRows);
  ASSERT_EQ(full->NumBatches(), 4);
  auto full_cuts = GetCutValues(&ctx, full.get(), kBins);
  ASSERT_FALSE(full_cuts.empty());

  for (std::int32_t f = 0; f < kFolds; ++f) {
    auto baseline = cv_test::MakeFoldBaseline(&ctx, data, folds, f, kBins, /*on_host=*/false,
                                              tmpdir.Str() + "/fold" + std::to_string(f), full);
    EXPECT_EQ(baseline->Info().num_row_, folds.TrainRows(f));
    // Sharing cuts via `ref` means the cut values must be identical.
    auto fold_cuts = GetCutValues(&ctx, baseline.get(), kBins);
    EXPECT_EQ(fold_cuts, full_cuts);
  }
}

namespace {
// Run the fused maker for `K` folds and return the grown trees, leaving `maker`'s per-fold
// state intact for follow-up queries (e.g. fetch counting is done by the caller).
void RunEquivalence(Context const* ctx, bst_idx_t n_rows, bst_feature_t n_cols, std::int32_t k,
                    bst_idx_t n_batches, bool on_host, bst_bin_t bins, std::int32_t max_depth) {
  auto data = cv_test::MakeCVTestData(n_rows, n_cols);
  auto folds = CVFoldInfo::MakeContiguous(n_rows, k);
  auto global_g = MakeGlobalGradients(data);

  common::TemporaryDirectory tmpdir;
  auto full = cv_test::MakeExtMemQdm(ctx, data, n_batches, bins, on_host, tmpdir.Str() + "/full");
  auto setup = MakeSharedSetup(ctx, full.get());
  auto param = MakeTrainParam(bins, max_depth);
  HistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(Args{});

  GPUFusedCVHistMaker maker{ctx,          param,       &hist_param, folds, setup.batch_ptr,
                            setup.cuts,   setup.dense, n_cols};

  // Per-fold global-sized gradients (validation rows zeroed), trees, and positions.
  std::vector<HostDeviceVector<GradientPair>> fgpair(k);
  std::vector<RegTree> ftrees(k);
  std::vector<HostDeviceVector<bst_node_t>> fpos(k);
  std::vector<HostDeviceVector<GradientPair>*> gptr;
  std::vector<RegTree*> tptr;
  std::vector<HostDeviceVector<bst_node_t>*> pptr;
  for (std::int32_t f = 0; f < k; ++f) {
    std::vector<GradientPair> hv = global_g;
    for (bst_idx_t i = folds.valid_ptr[f]; i < folds.valid_ptr[f + 1]; ++i) {
      hv[i] = GradientPair{0.0f, 0.0f};
    }
    fgpair[f].Resize(n_rows);
    fgpair[f].HostVector() = hv;
    fgpair[f].SetDevice(ctx->Device());
    gptr.push_back(&fgpair[f]);
    tptr.push_back(&ftrees[f]);
    pptr.push_back(&fpos[f]);
  }

  maker.UpdateTrees(full.get(), gptr, tptr, pptr);

  // Compare each fused tree to a baseline trained on a standalone shared-`ref` fold matrix.
  for (std::int32_t f = 0; f < k; ++f) {
    auto baseline = cv_test::MakeFoldBaseline(ctx, data, folds, f, bins, on_host,
                                              tmpdir.Str() + "/fold" + std::to_string(f), full);
    std::vector<GradientPair> bgpair;
    for (auto const& r : folds.TrainRanges(f)) {
      for (bst_idx_t i = r.first; i < r.second; ++i) {
        bgpair.push_back(global_g[i]);
      }
    }
    ASSERT_EQ(bgpair.size(), folds.TrainRows(f));
    auto btree = TrainBaselineTree(ctx, param, bgpair, baseline.get());
    EXPECT_TRUE(ftrees[f] == btree)
        << "Fused tree for fold " << f << " differs from the baseline (K=" << k
        << ", n_batches=" << n_batches << ").";
  }
}
}  // anonymous namespace

// Each fold's fused tree must be bit-identical to a baseline `grow_gpu_hist` tree trained on
// the fold's standalone shared-`ref` ExtMem matrix (success criterion #2).
TEST(GpuFusedCV, EquivalenceSingleBatch) {
  auto ctx = MakeCUDACtx(0);
  for (std::int32_t k : {2, 3, 5}) {
    RunEquivalence(&ctx, /*n_rows=*/600, /*n_cols=*/6, k, /*n_batches=*/1, /*on_host=*/false,
                   /*bins=*/32, /*max_depth=*/3);
  }
}

TEST(GpuFusedCV, EquivalenceMultiBatch) {
  auto ctx = MakeCUDACtx(0);
  // 4 batches of 150 rows. K=4 places a whole page inside each validation block.
  for (std::int32_t k : {3, 4}) {
    RunEquivalence(&ctx, /*n_rows=*/600, /*n_cols=*/6, k, /*n_batches=*/4, /*on_host=*/false,
                   /*bins=*/32, /*max_depth=*/3);
  }
}

TEST(GpuFusedCV, EquivalenceFoldNotDividingRows) {
  auto ctx = MakeCUDACtx(0);
  // 700 rows / 3 folds does not divide evenly, exercising the per-fold quantiser row count
  // (review #2 R2-A). 5 batches of 140 rows.
  RunEquivalence(&ctx, /*n_rows=*/700, /*n_cols=*/8, /*k=*/3, /*n_batches=*/5, /*on_host=*/false,
                 /*bins=*/48, /*max_depth=*/3);
}

TEST(GpuFusedCV, EquivalenceOnHost) {
  auto ctx = MakeCUDACtx(0);
  RunEquivalence(&ctx, /*n_rows=*/512, /*n_cols=*/6, /*k=*/4, /*n_batches=*/4, /*on_host=*/true,
                 /*bins=*/32, /*max_depth=*/3);
}

// Each source Ellpack page is fetched once per tree level for *all* folds combined, never
// once per fold (success criterion #3). Verified by asserting the per-`UpdateTrees` fetch
// delta is independent of the fold count K.
TEST(GpuFusedCV, PageReuse) {
  auto ctx = MakeCUDACtx(0);
  bst_idx_t constexpr kRows = 600;
  bst_feature_t constexpr kCols = 6;
  bst_bin_t constexpr kBins = 32;
  bst_idx_t constexpr kBatches = 4;
  std::int32_t constexpr kDepth = 3;

  auto data = cv_test::MakeCVTestData(kRows, kCols);
  auto global_g = MakeGlobalGradients(data);

  auto fetch_delta = [&](std::int32_t k) -> bst_idx_t {
    auto folds = CVFoldInfo::MakeContiguous(kRows, k);
    common::TemporaryDirectory tmpdir;
    auto full =
        cv_test::MakeExtMemQdm(&ctx, data, kBatches, kBins, /*on_host=*/false, tmpdir.Str() + "/f");
    auto setup = MakeSharedSetup(&ctx, full.get());
    auto param = MakeTrainParam(kBins, kDepth);
    HistMakerTrainParam hist_param;
    hist_param.UpdateAllowUnknown(Args{});
    GPUFusedCVHistMaker maker{&ctx,        param,       &hist_param, folds, setup.batch_ptr,
                              setup.cuts,  setup.dense, kCols};

    std::vector<HostDeviceVector<GradientPair>> fgpair(k);
    std::vector<RegTree> ftrees(k);
    std::vector<HostDeviceVector<bst_node_t>> fpos(k);
    std::vector<HostDeviceVector<GradientPair>*> gptr;
    std::vector<RegTree*> tptr;
    std::vector<HostDeviceVector<bst_node_t>*> pptr;
    for (std::int32_t f = 0; f < k; ++f) {
      std::vector<GradientPair> hv = global_g;
      for (bst_idx_t i = folds.valid_ptr[f]; i < folds.valid_ptr[f + 1]; ++i) {
        hv[i] = GradientPair{0.0f, 0.0f};
      }
      fgpair[f].Resize(kRows);
      fgpair[f].HostVector() = hv;
      fgpair[f].SetDevice(ctx.Device());
      gptr.push_back(&fgpair[f]);
      tptr.push_back(&ftrees[f]);
      pptr.push_back(&fpos[f]);
    }

    auto* ext = dynamic_cast<data::ExtMemQuantileDMatrix*>(full.get());
    CHECK(ext);
    auto before = ext->EllpackFetchCount();
    maker.UpdateTrees(full.get(), gptr, tptr, pptr);
    auto after = ext->EllpackFetchCount();
    return after - before;
  };

  auto delta2 = fetch_delta(2);
  auto delta4 = fetch_delta(4);
  // Fetch count is independent of the number of folds: pages are shared across folds.
  EXPECT_EQ(delta2, delta4);
  // And it is bounded by one fetch per page per level (root + at most `max_depth` levels),
  // ruling out any per-fold fetching (which would scale with K).
  EXPECT_GE(delta4, kBatches);
  EXPECT_LE(delta4, static_cast<bst_idx_t>(kDepth + 1) * kBatches);
}

namespace {
// Independent reference: train fold `f` on its own standalone matrix with the public
// `grow_gpu_hist` updater, maintaining the fold's training/validation margins with the same
// objective and the same binned predictor as the fused path, and return the per-round
// validation RMSE. This is the single-fold, separate-matrix analogue of `TrainFusedCV`.
std::vector<double> ReferenceFoldRmse(Context const* ctx, cv_test::CVTestData const& data,
                                      CVFoldInfo const& folds, std::int32_t f, bst_bin_t bins,
                                      bool on_host, std::string const& prefix,
                                      std::shared_ptr<DMatrix> full, TrainParam const& param,
                                      std::int32_t num_round) {
  auto train_fmat =
      cv_test::MakeFoldBaseline(ctx, data, folds, f, bins, on_host, prefix + "-tr", full);
  auto valid_fmat =
      cv_test::MakeFoldValidation(ctx, data, folds, f, bins, on_host, prefix + "-va", full);
  bst_idx_t n_train = folds.TrainRows(f);
  bst_idx_t n_valid = folds.ValidRows(f);

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("reg:squarederror", ctx)};
  obj->Configure(Args{});

  linalg::Vector<float> base_score;
  obj->InitEstimation(train_fmat->Info(), &base_score);
  obj->ProbToMargin(&base_score);
  float intercept = base_score.HostView()(0);

  HostDeviceVector<float> train_margin;
  train_margin.SetDevice(ctx->Device());
  train_margin.Resize(n_train, intercept);
  HostDeviceVector<float> valid_margin;
  valid_margin.SetDevice(ctx->Device());
  valid_margin.Resize(n_valid, intercept);

  auto const& valid_labels = valid_fmat->Info().labels.Data()->ConstHostVector();
  ObjInfo task{ObjInfo::kRegression};
  std::vector<double> rmse(num_round, 0.0);
  for (std::int32_t t = 0; t < num_round; ++t) {
    GradientContainer gc;
    gc.gpair = linalg::Matrix<GradientPair>{{n_train, bst_idx_t{1}}, ctx->Device()};
    obj->GetGradient(train_margin, train_fmat->Info(), t, &gc.gpair);

    std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_gpu_hist", ctx, &task)};
    updater->Configure(Args{});
    RegTree tree;
    std::vector<HostDeviceVector<bst_node_t>> position(1);
    updater->Update(&param, &gc, train_fmat.get(),
                    common::Span<HostDeviceVector<bst_node_t>>{position}, {&tree});

    PredictTreeBinned(ctx, train_fmat.get(), tree, train_margin.DeviceSpan());
    PredictTreeBinned(ctx, valid_fmat.get(), tree, valid_margin.DeviceSpan());

    auto const& h_valid = valid_margin.ConstHostVector();
    double sum_sq = 0.0;
    for (bst_idx_t i = 0; i < n_valid; ++i) {
      double d = static_cast<double>(h_valid[i]) - static_cast<double>(valid_labels[i]);
      sum_sq += d * d;
    }
    rmse[t] = std::sqrt(sum_sq / static_cast<double>(n_valid));
  }
  return rmse;
}

void RunTrainerEquivalence(Context const* ctx, bst_idx_t n_rows, bst_feature_t n_cols,
                           std::int32_t k, bst_idx_t n_batches, bool on_host, bst_bin_t bins,
                           std::int32_t max_depth, std::int32_t num_round) {
  auto data = cv_test::MakeCVTestData(n_rows, n_cols);
  auto folds = CVFoldInfo::MakeContiguous(n_rows, k);

  common::TemporaryDirectory tmpdir;
  auto full = cv_test::MakeExtMemQdm(ctx, data, n_batches, bins, on_host, tmpdir.Str() + "/full");
  auto param = MakeTrainParam(bins, max_depth);

  Args params{{"objective", "reg:squarederror"},
              {"max_depth", std::to_string(max_depth)},
              {"max_bin", std::to_string(bins)},
              {"min_child_weight", "0.0"},
              {"reg_alpha", "0"},
              {"reg_lambda", "0"},
              {"subsample", "1.0"},
              {"colsample_bynode", "1.0"},
              {"colsample_bylevel", "1.0"},
              {"colsample_bytree", "1.0"},
              {"seed", "1994"}};
  auto results = TrainFusedCV(ctx, full.get(), folds, params, num_round, "rmse");

  ASSERT_EQ(results.num_boost_round, num_round);
  ASSERT_EQ(results.n_folds, k);
  ASSERT_EQ(static_cast<std::int32_t>(results.per_fold.size()), num_round);

  for (std::int32_t f = 0; f < k; ++f) {
    auto ref = ReferenceFoldRmse(ctx, data, folds, f, bins, on_host,
                                 tmpdir.Str() + "/fold" + std::to_string(f), full, param,
                                 num_round);
    for (std::int32_t t = 0; t < num_round; ++t) {
      EXPECT_NEAR(results.per_fold[t][f], ref[t], 1e-5)
          << "Fused vs reference RMSE mismatch (fold " << f << ", round " << t << ", K=" << k
          << ", batches=" << n_batches << ").";
    }
  }

  // The aggregate mean must match the per-fold mean, and RMSE should drop over rounds.
  for (std::int32_t t = 0; t < num_round; ++t) {
    double mean = 0.0;
    for (std::int32_t f = 0; f < k; ++f) {
      mean += results.per_fold[t][f];
    }
    mean /= static_cast<double>(k);
    EXPECT_NEAR(results.test_mean[t], mean, 1e-9);
  }
  EXPECT_LT(results.test_mean.back(), results.test_mean.front())
      << "Validation RMSE did not improve over boosting rounds.";
}
}  // anonymous namespace

// End-to-end fused CV: the per-fold validation RMSE history must match an independent
// per-fold reference trained on standalone matrices (Phase 3 exit criterion). Equality is
// expected (not just closeness) because the fused path and the reference share the
// objective, the binned predictor, and produce bit-identical per-fold trees.
TEST(GpuFusedCV, TrainerEquivalenceSingleBatch) {
  auto ctx = MakeCUDACtx(0);
  for (std::int32_t k : {2, 3, 5}) {
    RunTrainerEquivalence(&ctx, /*n_rows=*/600, /*n_cols=*/6, k, /*n_batches=*/1,
                          /*on_host=*/false, /*bins=*/32, /*max_depth=*/3, /*num_round=*/6);
  }
}

TEST(GpuFusedCV, TrainerEquivalenceMultiBatch) {
  auto ctx = MakeCUDACtx(0);
  for (std::int32_t k : {3, 4}) {
    RunTrainerEquivalence(&ctx, /*n_rows=*/600, /*n_cols=*/6, k, /*n_batches=*/4,
                          /*on_host=*/false, /*bins=*/32, /*max_depth=*/3, /*num_round=*/6);
  }
}

TEST(GpuFusedCV, TrainerEquivalenceFoldNotDividingRows) {
  auto ctx = MakeCUDACtx(0);
  RunTrainerEquivalence(&ctx, /*n_rows=*/700, /*n_cols=*/8, /*k=*/3, /*n_batches=*/5,
                        /*on_host=*/false, /*bins=*/48, /*max_depth=*/3, /*num_round=*/5);
}

TEST(GpuFusedCV, TrainerEquivalenceOnHost) {
  auto ctx = MakeCUDACtx(0);
  RunTrainerEquivalence(&ctx, /*n_rows=*/512, /*n_cols=*/6, /*k=*/4, /*n_batches=*/4,
                        /*on_host=*/true, /*bins=*/32, /*max_depth=*/3, /*num_round=*/5);
}
}  // namespace xgboost::tree
