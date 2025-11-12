/**
 * Copyright 2025, XGBoost contributors
 */
#include "../cross_validate/folds.h"
#include "../data/array_interface.h"
#include "c_api_error.h"
#include "xgboost/json.h"

using namespace xgboost;  // NOLINT


XGB_DLL int XGBCvUpdateOneIter(DMatrixHandle fmat, char const* tr_indices) {
  using BatchTrIdx = std::vector<std::vector<bst_idx_t>>;

  API_BEGIN();
  auto p_fmat = CastDMatrixHandle(fmat);
  CHECK(p_fmat);

  auto jindices = Json::Load(StringView{tr_indices});
  auto const& jindices_array = get<Array const>(jindices);
  std::size_t n_batches = jindices_array.size();
  std::cout << "n_batches:" << n_batches << std::endl;

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
  API_END();
}
