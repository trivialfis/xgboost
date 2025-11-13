/**
 * Copyright 2025, XGBoost contributors
 */
#include "../cross_validate/hist_build_trees.h"
#include "../data/array_interface.h"
#include "c_api_error.h"
#include "xgboost/json.h"

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvUpdateOneIter(DMatrixHandle fmat, char const* tr_indices, char const* grad,
                               char const* hess) {
  using BatchTrIdx = std::vector<std::vector<bst_idx_t>>;

  API_BEGIN();
  auto p_fmat = CastDMatrixHandle(fmat);
  CHECK(p_fmat);

  auto jindices = Json::Load(StringView{tr_indices});
  auto const& jindices_array = get<Array const>(jindices);
  std::size_t n_batches = jindices_array.size();

  std::int32_t n_folds = 0;
  std::vector<BatchTrIdx> tr_idx;
  for (std::size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    auto const& batch = get<Array const>(jindices[batch_idx]);
    if (n_folds == 0) {
      n_folds = batch.size();
    }
    CHECK_EQ(n_folds, batch.size());
    BatchTrIdx batch_tr_idx;
    for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto const& jfold = get<Object const>(batch[fold_idx]);
      auto fold = ArrayInterface<1>{jfold};
      batch_tr_idx.emplace_back();
      auto& fold_tr_idx = batch_tr_idx.back();
      DispatchDType(fold, DeviceOrd::CPU(), [&](auto&& in) {
        for (std::size_t i = 0; i < in.Shape(0); ++i) {
          fold_tr_idx.push_back(in(i));
        }
      });
    }
    CHECK_EQ(batch_tr_idx.size(), n_folds);
    tr_idx.emplace_back(std::move(batch_tr_idx));
  }
  CHECK_EQ(tr_idx.size(), n_batches);
  std::cout << "n_batches:" << n_batches << " n_folds:" << n_folds << std::endl;

  // Load gradient
  auto jgrad = Json::Load(grad);
  auto jhess = Json::Load(hess);

  auto const& jgrad_array = get<Array const>(jgrad);
  CHECK_EQ(jgrad_array.size(), n_batches);
  auto const& jhess_array = get<Array const>(jhess);
  CHECK_EQ(jhess_array.size(), n_batches);

  std::vector<std::vector<std::unique_ptr<GradientContainer>>> gpairs;
  for (std::size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    auto const& batch_grad = get<Array const>(jgrad_array[batch_idx]);
    CHECK_EQ(batch_grad.size(), n_folds);
    auto const& batch_hess = get<Array const>(jhess_array[batch_idx]);
    CHECK_EQ(batch_hess.size(), n_folds);
    std::vector<std::unique_ptr<GradientContainer>> batch_gpairs;
    for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto fold_grad = ArrayInterface<1>{get<Object const>(batch_grad[fold_idx])};
      auto fold_hess = ArrayInterface<1>{get<Object const>(batch_hess[fold_idx])};

      auto fold_gpair = std::make_unique<GradientContainer>();
      fold_gpair->gpair.Reshape(fold_grad.Shape<0>(), 1);
      auto& h_gpair = fold_gpair->gpair.Data()->HostVector();
      CHECK_EQ(h_gpair.size(), fold_grad.n);
      for (std::size_t i = 0; i < h_gpair.size(); ++i) {
        h_gpair[i] = GradientPair{fold_grad(i), fold_hess(i)};
      }
      batch_gpairs.emplace_back(std::move(fold_gpair));
    }
    CHECK_EQ(batch_gpairs.size(), n_folds);
    gpairs.emplace_back(std::move(batch_gpairs));
  }

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "cuda"}});

  std::vector<std::unique_ptr<RegTree>> trees;
  for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    trees.emplace_back(std::make_unique<RegTree>(1, p_fmat->Info().num_col_));
  }
  std::vector<RegTree*> p_trees;
  std::transform(trees.begin(), trees.end(), std::back_inserter(p_trees),
                 [](auto& t) { return t.get(); });

  cv::BuildTrees(&ctx, p_fmat.get(), gpairs, tr_idx, p_trees);

  API_END();
}
