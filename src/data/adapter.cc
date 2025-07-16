/**
 *  Copyright 2019-2025, XGBoost Contributors
 */
#include "adapter.h"

#include <utility>  // for move

#include "../c_api/c_api_error.h"  // for API_BEGIN, API_END
#include "array_interface.h"       // for ArrayInterface
#include "xgboost/c_api.h"
#include "../data/cat_container.h"
#include "xgboost/logging.h"

namespace xgboost::data {
ColumnarAdapter::ColumnarAdapter(StringView columns) {
  auto jarray = Json::Load(columns);
  CHECK(IsA<Array>(jarray));
  auto const& array = get<Array const>(jarray);
  bst_idx_t n_samples{0};
  std::vector<std::int32_t> cat_segments{0};
  for (auto const& jcol : array) {
    std::int32_t n_cats{0};
    if (IsA<Array>(jcol)) {
      // This is a dictionary type (categorical values).
      auto const& first = get<Object const>(jcol[0]);
      if (first.find("offsets") == first.cend()) {
        // numeric index
        n_cats = GetArrowNumericIndex(DeviceOrd::CPU(), jcol, &this->cats_, &this->columns_,
                                      &this->n_bytes_, &n_samples);
      } else {
        // string index
        n_cats =
            GetArrowDictionary(jcol, &this->cats_, &this->columns_, &this->n_bytes_, &n_samples);
        std::cout << "n_cats:" << n_cats << std::endl;
      }
    } else {
      // Numeric values
      columns_.emplace_back(get<Object const>(jcol));
      this->cats_.emplace_back();
      this->n_bytes_ += columns_.back().ElementSize() * columns_.back().Shape<0>();
      n_samples = std::max(n_samples, static_cast<bst_idx_t>(columns_.back().Shape<0>()));
    }
    cat_segments.push_back(n_cats);
  }
  std::partial_sum(cat_segments.cbegin(), cat_segments.cend(), cat_segments.begin());
  auto no_overflow = std::is_sorted(cat_segments.cbegin(), cat_segments.cend());
  CHECK(no_overflow) << "Maximum number of categories exceeded.";

  // Check consistency.
  bool consistent = columns_.empty() || std::all_of(columns_.cbegin(), columns_.cend(),
                                                    [&](ArrayInterface<1> const& array) {
                                                      return array.Shape<0>() == n_samples;
                                                    });
  this->cat_segments_ = std::move(cat_segments);
  CHECK(consistent) << "Size of columns should be the same.";
  batch_ = ColumnarAdapterBatch{columns_};
  for (auto const& col : this->cats_) {
    std::visit(enc::Overloaded{enc::MakeCatStrViewOp([](auto const& str) {
                                 std::cout << "Offset:" << std::endl;
                                 for (std::size_t i = 0; i < str.offsets.size(); ++i) {
                                   std::cout << str.offsets[i] << ", ";
                                 }
                                 for (std::size_t i = 1; i < str.offsets.size(); ++i) {
                                   auto begin = str.offsets[i - 1];
                                   auto end = str.offsets[i];
                                   std::cout << "begin:" << begin << " end:" << end << std::endl;
                                   for (std::int32_t j = begin; j < end; ++j) {
                                     std::cout << str.values[j];
                                   }
                                   std::cout << std::endl;
                                 }
                               }),
                               [](auto&&) {

                               }},

               col);
  }
}

template <typename DataIterHandle, typename XGBCallbackDataIterNext, typename XGBoostBatchCSR>
bool IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>::Next() {
  if ((*next_callback_)(
          data_handle_,
          [](void* handle, XGBoostBatchCSR batch) -> int {
            API_BEGIN();
            static_cast<IteratorAdapter*>(handle)->SetData(batch);
            API_END();
          },
          this) != 0) {
    at_first_ = false;
    return true;
  } else {
    return false;
  }
}

template class IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext, XGBoostBatchCSR>;
}  // namespace xgboost::data
