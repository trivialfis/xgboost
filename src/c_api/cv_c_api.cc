/**
 * Copyright 2025, XGBoost contributors
 */
#include "../cross_validate/folds.h"
#include "../data/array_interface.h"
#include "c_api_error.h"
#include "xgboost/json.h"

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvSetIndex(char const* indices) {
  API_BEGIN();
  auto jindices = Json::Load(StringView{indices});
  auto const& jindices_array = get<Array const>(jindices);
  std::size_t n_folds = jindices_array.size();
  std::int32_t n_batches = -1;
  std::vector<std::vector<std::vector<cv::Segment>>> segments;
  for (std::size_t f = 0; f < n_folds; ++f) {
    auto const& jfold = jindices_array[f];
    auto const& fold = get<Array const>(jfold);
    if (n_batches == -1) {
      n_batches = fold.size();
    }
    CHECK_EQ(n_batches, fold.size())
        << "The number of batches must be consistent across all folds.";

    std::vector<std::vector<cv::Segment>> fold_segments;

    for (std::int32_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
      // integer array for index, encoded as array intereface
      auto const& jbatch = get<Object const>(fold[batch_idx]);
      auto array = ArrayInterface<1>{jbatch};
      std::vector<bst_idx_t> pos{0};
      std::vector<bst_idx_t> cnt{1};  // first sample
      CHECK_EQ(array.type, xgboost::ArrayInterfaceHandler::kI8);
      auto t = linalg::TensorView<std::int64_t const, 1>{
          common::Span{static_cast<std::int64_t const*>(array.data),
                       std::numeric_limits<std::size_t>::max()},
          array.shape, array.strides, DeviceOrd::CPU()};

      auto n_samples = t.Shape(0);
      CHECK_GE(n_samples, 1);

      for (std::size_t i = 1; i < n_samples; ++i) {
        auto diff = t(i) - t(i - 1);
        if (diff > 1) {
          pos.push_back(i);
          cnt.push_back(1);
        } else {
          cnt[pos.size() - 1]++;
        }
      }

      // Segments within one batch of one fold
      std::vector<cv::Segment> batch_segments;
      CHECK_EQ(pos.size(), cnt.size());
      for (std::size_t k = 0; k < pos.size(); ++k) {
        batch_segments.emplace_back(pos[k], cnt[k]);
      }
      fold_segments.emplace_back(std::move(batch_segments));
    }
    CHECK_EQ(fold_segments.size(), n_batches);
    segments.emplace_back(std::move(fold_segments));
  }
  API_END();
}
