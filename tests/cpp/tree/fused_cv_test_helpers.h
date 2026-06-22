/**
 * Copyright 2026, XGBoost Contributors
 *
 * @brief Test helpers for the fused cross-validation training path.
 *
 * These helpers build the shared, full `ExtMemQuantileDMatrix` over a deterministic
 * synthetic dataset, and a standalone single-page `ExtMemQuantileDMatrix` containing exactly
 * a fold's training rows that shares cuts with the full matrix via the `ref` constructor.
 * They are CUDA-aware (the iterators emit array interfaces) and therefore intended to be
 * included from `.cu` test translation units.
 */
#ifndef XGBOOST_TESTS_CPP_TREE_FUSED_CV_TEST_HELPERS_H_
#define XGBOOST_TESTS_CPP_TREE_FUSED_CV_TEST_HELPERS_H_

#include <xgboost/context.h>             // for Context
#include <xgboost/data.h>                // for DMatrix, ExtMemConfig
#include <xgboost/host_device_vector.h>  // for HostDeviceVector

#include <limits>   // for numeric_limits
#include <memory>   // for shared_ptr, unique_ptr
#include <random>   // for mt19937_64
#include <string>   // for string
#include <utility>  // for move
#include <vector>   // for vector

#include "../../../src/data/batch_utils.h"  // for AutoHostRatio, AutoCachePageBytes
#include "../../../src/tree/cv_fold_info.h"  // for CVFoldInfo, RowRange
#include "../helpers.h"                      // for ArrayIterForTest, Reset, Next

namespace xgboost::tree::cv_test {
/** @brief A deterministic in-memory dense dataset (row-major). */
struct CVTestData {
  bst_idx_t n_rows{0};
  bst_feature_t n_features{0};
  std::vector<float> values;  // size n_rows * n_features, row-major
  std::vector<float> labels;  // size n_rows
};

/**
 * @brief Generate a deterministic dense dataset with regression-style labels.
 */
inline CVTestData MakeCVTestData(bst_idx_t n_rows, bst_feature_t n_features,
                                 std::uint64_t seed = 1994) {
  CVTestData d;
  d.n_rows = n_rows;
  d.n_features = n_features;
  d.values.resize(static_cast<std::size_t>(n_rows) * n_features);
  d.labels.resize(n_rows);
  std::mt19937_64 rng{seed};
  std::uniform_real_distribution<float> feat{-2.0f, 2.0f};
  std::normal_distribution<float> noise{0.0f, 0.1f};
  for (bst_idx_t i = 0; i < n_rows; ++i) {
    float y = 0.0f;
    for (bst_feature_t j = 0; j < n_features; ++j) {
      auto v = feat(rng);
      d.values[static_cast<std::size_t>(i) * n_features + j] = v;
      y += (j % 2 == 0 ? 1.0f : -0.5f) * v;
    }
    d.labels[i] = y + noise(rng);
  }
  return d;
}

/** @brief Build a new dataset from the rows covered by `ranges` (in order). */
inline CVTestData SubsetRows(CVTestData const& full, std::vector<RowRange> const& ranges) {
  CVTestData d;
  d.n_features = full.n_features;
  for (auto const& r : ranges) {
    for (bst_idx_t i = r.first; i < r.second; ++i) {
      for (bst_feature_t j = 0; j < full.n_features; ++j) {
        d.values.push_back(full.values[static_cast<std::size_t>(i) * full.n_features + j]);
      }
      d.labels.push_back(full.labels[i]);
    }
  }
  d.n_rows = d.labels.size();
  return d;
}

/**
 * @brief Build an `ExtMemQuantileDMatrix` from a dataset split into `n_batches` equal pages.
 *
 * @param ref Optional reference matrix to share cuts with (pass `nullptr` for the full
 *            matrix). The fold matrices pass the full matrix here.
 */
inline std::shared_ptr<DMatrix> MakeExtMemQdm(Context const* ctx, CVTestData const& d,
                                              bst_idx_t n_batches, bst_bin_t bins, bool on_host,
                                              std::string const& prefix,
                                              std::shared_ptr<DMatrix> ref = nullptr) {
  CHECK_GE(n_batches, 1);
  CHECK_EQ(d.n_rows % n_batches, 0) << "Test rows must divide evenly into batches.";
  bst_idx_t rows_per_batch = d.n_rows / n_batches;

  HostDeviceVector<float> data;
  data.Resize(d.values.size());
  data.HostVector() = d.values;
  if (ctx->IsCUDA()) {
    data.SetDevice(ctx->Device());
    data.ConstDevicePointer();
  }

  std::unique_ptr<ArrayIterForTest> iter;
  if (ctx->IsCPU()) {
    iter = std::make_unique<NumpyArrayIterForTest>(ctx, data, rows_per_batch, d.n_features,
                                                   n_batches);
  } else {
    iter = std::make_unique<CudaArrayIterForTest>(ctx, data, rows_per_batch, d.n_features,
                                                  n_batches);
  }

  ExtMemConfig config{
      prefix,
      on_host,
      ::xgboost::cuda_impl::AutoHostRatio(),
      ::xgboost::cuda_impl::AutoCachePageBytes(),
      std::numeric_limits<float>::quiet_NaN(),
      ctx->Threads(),
  };
  std::shared_ptr<DMatrix> p_fmat{DMatrix::Create(static_cast<DataIterHandle>(iter.get()),
                                                  iter->Proxy(), std::move(ref), Reset, Next, bins,
                                                  config)};
  // Attach labels.
  auto& labels = p_fmat->Info().labels;
  labels.Reshape(d.n_rows, 1);
  labels.Data()->HostVector() = d.labels;
  if (ctx->IsCUDA()) {
    labels.SetDevice(ctx->Device());
    labels.Data()->ConstDevicePointer();
  }
  return p_fmat;
}

/**
 * @brief Convenience: build the per-fold baseline standalone matrix (single page) for fold
 *        `f`, sharing cuts with `full` via `ref`.
 */
inline std::shared_ptr<DMatrix> MakeFoldBaseline(Context const* ctx, CVTestData const& data,
                                                 CVFoldInfo const& folds, std::int32_t f,
                                                 bst_bin_t bins, bool on_host,
                                                 std::string const& prefix,
                                                 std::shared_ptr<DMatrix> full) {
  auto sub = SubsetRows(data, folds.TrainRanges(f));
  return MakeExtMemQdm(ctx, sub, /*n_batches=*/1, bins, on_host, prefix, std::move(full));
}

/**
 * @brief Convenience: build the per-fold standalone **validation** matrix (single page) for
 *        fold `f`, sharing cuts with `full` via `ref`.
 */
inline std::shared_ptr<DMatrix> MakeFoldValidation(Context const* ctx, CVTestData const& data,
                                                   CVFoldInfo const& folds, std::int32_t f,
                                                   bst_bin_t bins, bool on_host,
                                                   std::string const& prefix,
                                                   std::shared_ptr<DMatrix> full) {
  auto sub = SubsetRows(data, folds.ValidRanges(f));
  return MakeExtMemQdm(ctx, sub, /*n_batches=*/1, bins, on_host, prefix, std::move(full));
}
}  // namespace xgboost::tree::cv_test

#endif  // XGBOOST_TESTS_CPP_TREE_FUSED_CV_TEST_HELPERS_H_
