/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <cmath>    // for abs
#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
#include <string>   // for to_string
#include <vector>   // for vector

#include "../../../src/common/random.h"                  // for ColumnSampler
#include "../../../src/cross_validate/cross_validate.h"  // for FoldInfoBatches, FoldGpairs
#include "../../../src/cross_validate/kfolds.h"          // for KFold
#include "../../../src/tree/hist/hist_param.h"           // for HistMakerTrainParam
#include "../../../src/tree/param.h"                     // for TrainParam, CalcWeight
#include "../../../src/tree/updater_gpu_common.cuh"      // for HistBatch
#include "../../../src/tree/updater_gpu_cv_hist.cuh"     // for FusedCvHistTreeMaker
#include "../../../src/tree/updater_gpu_hist.cuh"        // for InitBatchCuts
#include "../helpers.h"                                  // for RandomDataGenerator, MakeCUDACtx

namespace xgboost::tree::cuda_impl {
namespace {
struct CvRootTestData {
  cv::FoldInfoBatches finfo;
  cv::FoldGpairs gpairs;
  std::vector<bst_idx_t> batch_ptr;
  // Expected root sums, indexed by [fold][target].
  std::vector<std::vector<GradientPairPrecise>> root_sum;
};

[[nodiscard]] CvRootTestData MakeCvRootData(Context const* ctx, DMatrix* p_fmat,
                                            std::size_t k_folds, bst_target_t n_targets) {
  CvRootTestData out;
  auto batch_ptr = p_fmat->BatchPtr();
  auto n_batches = batch_ptr.size() - 1;
  out.batch_ptr = batch_ptr;
  out.root_sum.resize(k_folds, std::vector<GradientPairPrecise>(n_targets));

  for (std::size_t i = 0; i < n_batches; ++i) {
    out.finfo.batches.emplace_back();
    auto& batch = out.finfo.batches.back();
    for (std::size_t k = 0; k < k_folds; ++k) {
      batch.ridxs.emplace_back();
      cv::KFold(ctx, k_folds, batch_ptr[i], batch_ptr[i + 1], static_cast<std::int32_t>(k),
                &batch.ridxs.back());
    }
  }

  // reg:squarederror with zero initial prediction and unit weights:
  // grad = -label, hess = 1.
  auto h_labels = p_fmat->Info().labels.HostView();
  out.gpairs.gpairs.resize(k_folds);
  for (std::size_t k = 0; k < k_folds; ++k) {
    std::vector<GradientPair> h_g;
    for (std::size_t i = 0; i < n_batches; ++i) {
      for (auto r : out.finfo.batches[i].ridxs[k].HostVector()) {
        auto global = r + batch_ptr[i];
        for (bst_target_t t = 0; t < n_targets; ++t) {
          auto y = h_labels(global, t);
          auto g = GradientPair{-y, 1.0f};
          h_g.emplace_back(g);
          out.root_sum[k][t] += GradientPairPrecise{g.GetGrad(), g.GetHess()};
        }
      }
    }
    auto fold_size = h_g.size() / n_targets;
    out.gpairs.gpairs[k].Reshape(fold_size, n_targets);
    if (!h_g.empty()) {
      out.gpairs.gpairs[k].Data()->HostVector() = h_g;
    }
  }
  return out;
}

std::vector<RegTree*> MakeTreePtrs(std::vector<RegTree>* p_trees, std::size_t k_folds,
                                   bst_target_t n_targets, bst_feature_t n_features) {
  auto& trees = *p_trees;
  trees.clear();
  trees.reserve(k_folds);
  std::vector<RegTree*> ptrs;
  ptrs.reserve(k_folds);
  for (std::size_t k = 0; k < k_folds; ++k) {
    trees.emplace_back(n_targets, n_features, true);
    CHECK(trees.back().IsMultiTarget());
    ptrs.push_back(&trees.back());
  }
  return ptrs;
}

void CheckPartitioners(FusedCvHistTreeMaker* maker, CvRootTestData const& data) {
  for (std::size_t k = 0; k < maker->KFolds(); ++k) {
    for (std::size_t i = 0; i + 1 < data.batch_ptr.size(); ++i) {
      auto got = maker->Partitioners(k).At(i)->GetRowsHost(RegTree::kRoot);
      std::vector<RowIndexT> expected;
      for (auto r : data.finfo.batches[i].ridxs[k].HostVector()) {
        expected.push_back(static_cast<RowIndexT>(r + data.batch_ptr[i]));
      }
      ASSERT_EQ(got, expected) << "fold=" << k << " batch=" << i;
    }
  }
}

void CheckRootSums(FusedCvHistTreeMaker* maker, CvRootTestData const& data,
                   bst_target_t n_targets) {
  for (std::size_t k = 0; k < maker->KFolds(); ++k) {
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto got = maker->RootSum(k, t);
      auto want = data.root_sum[k][t];
      EXPECT_NEAR(got.GetGrad(), want.GetGrad(), 1e-2 * (std::abs(want.GetGrad()) + 1.0))
          << "fold=" << k << " target=" << t;
      EXPECT_NEAR(got.GetHess(), want.GetHess(), 1e-2 * (want.GetHess() + 1.0))
          << "fold=" << k << " target=" << t;
    }
  }
}

void CheckRootLeaves(FusedCvHistTreeMaker* maker, TrainParam const& param,
                     std::vector<RegTree>* p_trees, bst_target_t n_targets) {
  for (std::size_t k = 0; k < maker->KFolds(); ++k) {
    auto* mt = p_trees->at(k).GetMultiTargetTree();
    mt->SetLeaves();
    auto leaf = mt->LeafValue(RegTree::kRoot);
    ASSERT_EQ(leaf.Size(), n_targets);
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto want = param.learning_rate * CalcWeight(param, maker->RootSum(k, t));
      EXPECT_NEAR(leaf(t), want, 1e-5) << "fold=" << k << " target=" << t;
    }
  }
}

void RunFusedCvRoot(std::size_t k_folds, std::size_t n_batches, bst_target_t n_targets) {
  auto ctx = MakeCUDACtx(0);
  bst_idx_t n_samples = 512;
  bst_feature_t n_features = 8;
  bst_bin_t max_bin = 16;

  auto p_fmat = RandomDataGenerator{n_samples, n_features, 0.0f}
                    .Batches(n_batches)
                    .Bins(max_bin)
                    .Targets(n_targets)
                    .Device(ctx.Device())
                    .GenerateExtMemQuantileDMatrix("temp", true);
  auto& info = p_fmat->Info();
  ASSERT_EQ(info.labels.Shape(1), n_targets);

  auto data = MakeCvRootData(&ctx, p_fmat.get(), k_folds, n_targets);

  tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_bin", std::to_string(max_bin)}});
  HistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(Args{});

  auto batch = HistBatch(param);
  auto [cuts, dense_compressed] = InitBatchCuts(&ctx, p_fmat.get(), batch);
  std::vector<std::shared_ptr<common::ColumnSampler>> col_samplers;
  for (std::size_t i = 0; i < k_folds; ++i) {
    col_samplers.emplace_back(std::make_shared<common::ColumnSampler>());
  }

  FusedCvHistTreeMaker maker{&ctx,           param, &hist_param,     col_samplers,
                             data.batch_ptr, cuts,  dense_compressed};

  std::vector<RegTree> trees;
  auto tree_ptrs = MakeTreePtrs(&trees, k_folds, n_targets, n_features);

  maker.Reset(p_fmat.get(), data.finfo, data.gpairs);
  ASSERT_EQ(maker.NTargets(), n_targets);
  auto entries = maker.InitRoots(p_fmat.get(), tree_ptrs);
  ASSERT_EQ(entries.size(), k_folds);

  auto total_bins = cuts->TotalBins();
  CheckPartitioners(&maker, data);
  CheckRootSums(&maker, data, n_targets);

  for (std::size_t k = 0; k < k_folds; ++k) {
    ASSERT_EQ(maker.RootHistogram(k).size(), static_cast<std::size_t>(n_targets) * total_bins);

    auto const& e = entries[k];
    if (e.IsValid(param, 1)) {
      EXPECT_GE(e.split.findex, 0);
      EXPECT_LT(e.split.findex, static_cast<int>(n_features));
      EXPECT_GE(e.split.loss_chg, 0.0f);
    }
  }

  CheckRootLeaves(&maker, param, &trees, n_targets);
}
}  // namespace

// Single-target is the trivial n_targets == 1 case flowing through the vector-leaf path.
TEST(GpuFusedCvRoot, SingleBatch) { RunFusedCvRoot(3, 1, 1); }

TEST(GpuFusedCvRoot, MultiBatch) { RunFusedCvRoot(3, 4, 1); }

TEST(GpuFusedCvRoot, MoreFolds) { RunFusedCvRoot(5, 3, 1); }

// k_folds == 1 is degenerate: every batch has zero training rows for the single fold,
// exercising the empty-per-batch guard in the fused root build.
TEST(GpuFusedCvRoot, EmptyTraining) { RunFusedCvRoot(1, 2, 1); }

// Multi-target (vector leaf) cases.
TEST(GpuFusedCvRoot, MultiTargetSingleBatch) { RunFusedCvRoot(3, 1, 3); }

TEST(GpuFusedCvRoot, MultiTargetMultiBatch) { RunFusedCvRoot(3, 4, 3); }

TEST(GpuFusedCvRoot, MultiTargetMoreFolds) { RunFusedCvRoot(4, 3, 2); }
}  // namespace xgboost::tree::cuda_impl
