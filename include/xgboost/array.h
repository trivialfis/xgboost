#ifndef XGBOOST_ARRAY_H
#define XGBOOST_ARRAY_H

#include "./logging.h"
#include "./span.h"

#include <cstring>

#include <array>
#include <vector>
#include <limits>

namespace xgboost {

template <typename T>
class ArrayView {
  common::Span<T> data_;
  common::Span<uint64_t, 4> shape_;
  common::Span<uint64_t, 4> strides_;

 public:
  static constexpr uint64_t kAll = std::numeric_limits<uint64_t>::max();

  ArrayView(common::Span<T> data, common::Span<T> shape, common::Span<T> strides) :
      data_{data}, shape_{shape}, strides_{strides} {}
  T const* Data() const { return data_.data(); }
  T* Data() { return data_.data(); }

  ArrayView<T> Slice(std::vector<uint64_t> const& index);
  bool operator==(ArrayView<T> const& that);

  ArrayView<T>& Copy(ArrayView<T> const& that) {
    CHECK_EQ(data_.size(), that.data_.size());
    std::memcpy(data_.data(), that.data_.data(), sizeof(T) * data_.size());

    CHECK_EQ(shape_.size(), that.shape_.size());
    std::memcpy(shape_.data(), that.shape_.data());

    CHECK_EQ(strides_.size(), that.strides_.size());
    std::memcpy(strides_.data(), that.strides_.data());
  }
};

}      // namespace xgboost

#endif  // XGBOOST_ARRAY_H
