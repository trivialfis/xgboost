/**
 * Copyright 2024-2025, XGBoost contributors
 */
#pragma once

#include <variant>  // for visit
#include <vector>   // for vector

#include "../../../src/encoder/types.h"        // for Overloaded
#include "../../src/common/device_vector.cuh"  // for device_vector
#include "df_mock.h"                           // for MakeStrArrayImpl

namespace enc::cuda_impl {
struct CatStrArray {
  dh::device_vector<std::int32_t> offsets;
  dh::device_vector<CatCharT> values;

  CatStrArray() = default;
  CatStrArray(CatStrArray const& that) = delete;
  CatStrArray& operator=(CatStrArray const& that) = delete;

  CatStrArray(CatStrArray&& that) = default;
  CatStrArray& operator=(CatStrArray&& that) = default;

  [[nodiscard]] explicit operator CatStrArrayView() const {
    return {dh::ToSpan(offsets), dh::ToSpan(values)};
  }
  [[nodiscard]] std::size_t size() const {  // NOLINT
    return CatStrArrayView(*this).size();
  }

  void Copy(CatStrArray const& that) {
    this->offsets = that.offsets;
    this->values = that.values;
  }
};

template <typename T>
struct ViewToStorageImpl;

template <>
struct ViewToStorageImpl<CatStrArrayView> {
  using Type = CatStrArray;
};

template <typename T>
struct ViewToStorageImpl<::xgboost::common::Span<T const>> {
  using Type = dh::device_vector<T>;
};

template <typename... Ts>
struct ViewToStorage;

template <typename... Ts>
struct ViewToStorage<std::tuple<Ts...>> {
  using Type = std::tuple<typename ViewToStorageImpl<Ts>::Type...>;
};

using CatIndexTypes = ViewToStorage<CatIndexViewTypes>::Type;

using ColumnType = cpu_impl::TupToVarT<CatIndexTypes>;

class DfTest {
 public:
  template <typename T>
  using Vector = dh::device_vector<T>;

 private:
  std::vector<ColumnType> columns_;
  dh::device_vector<DeviceCatIndexView> columns_v_;
  dh::device_vector<std::int32_t> segments_;
  std::vector<std::int32_t> h_segments_;

  dh::device_vector<std::int32_t> mapping_;

  template <typename Head>
  static void MakeImpl(std::vector<ColumnType>* p_out, dh::device_vector<std::int32_t>* p_sizes,
                       Head&& col) {
    p_sizes->push_back(col.size());
    p_out->emplace_back(std::forward<Head>(col));

    p_sizes->insert(p_sizes->begin(), 0);
    thrust::inclusive_scan(p_sizes->cbegin(), p_sizes->cend(), p_sizes->begin());
  }

  template <typename Head, typename... Col>
  static void MakeImpl(std::vector<ColumnType>* p_out, dh::device_vector<std::int32_t>* p_sizes,
                       Head&& col, Col&&... columns) {
    p_sizes->push_back(col.size());
    p_out->emplace_back(std::forward<Head>(col));
    MakeImpl(p_out, p_sizes, std::forward<Col>(columns)...);
  }

 public:
  template <typename... Col>
  static DfTest Make(Col&&... columns) {
    DfTest df;
    MakeImpl(&df.columns_, &df.segments_, std::forward<Col>(columns)...);
    for (std::size_t i = 0; i < df.columns_.size(); ++i) {
      auto const& col = df.columns_[i];
      std::visit(
          Overloaded{[&](CatStrArray const& str) { df.columns_v_.push_back(CatStrArrayView(str)); },
                     [&](auto&& args) {
                       df.columns_v_.push_back(dh::ToSpan(args));
                     }},
          col);
    }
    CHECK_EQ(df.columns_v_.size(), sizeof...(columns));
    df.h_segments_.resize(df.segments_.size());
    thrust::copy_n(df.segments_.cbegin(), df.segments_.size(), df.h_segments_.begin());
    df.mapping_.resize(df.h_segments_.back());
    return df;
  }

  template <typename... Strs>
  static auto MakeStrs(Strs&&... strs) {
    auto array = MakeStrArrayImpl(std::forward<Strs>(strs)...);
    return CatStrArray{array.offsets, array.values};
  }

  template <typename... Ints>
  static auto MakeInts(Ints&&... names) {
    return dh::device_vector<std::int32_t>{names...};
  }

  auto View() const {
    return DeviceColumnsView{dh::ToSpan(this->columns_v_), dh::ToSpan(segments_),
                             h_segments_.back()};
  }
  auto Segment() const { return Span{h_segments_}; }

  auto MappingView() { return dh::ToSpan(mapping_); }
  auto const& Mapping() { return mapping_; }
};
}  // namespace enc::cuda_impl
