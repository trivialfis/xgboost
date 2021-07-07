/*!
 * Copyright 2020-2021 XGBoost contributors
 */
#ifndef XGBOOST_TESTS_CPP_DATA_ITERATOR_FOR_TESTS_H_
#define XGBOOST_TESTS_CPP_DATA_ITERATOR_FOR_TESTS_H_
#include <xgboost/host_device_vector.h>
#include <memory>
#include "helpers.h"

namespace xgboost {

typedef void *DMatrixHandle;  // NOLINT(*);
typedef void *DataIterHandle;  // NOLINT(*)

// An data iterator for testing external memory.
class ArrayIterForTest {
 protected:
  HostDeviceVector<float> data_;
  size_t iter_ {0};
  DMatrixHandle proxy_;
  std::unique_ptr<RandomDataGenerator> rng_;

  std::vector<std::string> batches_;
  std::string interface_;
  size_t rows_;
  size_t cols_;
  size_t n_batches_;

 public:
  size_t static constexpr kRows { 1000 };
  size_t static constexpr kBatches { 100 };
  size_t static constexpr kCols { 13 };

  std::string AsArray() const {
    return interface_;
  }

  virtual int Next();
  virtual void Reset() {
    iter_ = 0;
  }
  size_t Iter() const { return iter_; }
  auto Proxy() -> decltype(proxy_) { return proxy_; }

  explicit ArrayIterForTest(float sparsity, size_t rows = kRows,
                            size_t cols = kCols, size_t batches = kBatches);
  virtual ~ArrayIterForTest();
};

// A data iterator for testing device DMatrix.
class CudaArrayIterForTest : public ArrayIterForTest {
 public:
  size_t static constexpr kRows{1000};
  size_t static constexpr kBatches{100};
  size_t static constexpr kCols{13};

  explicit CudaArrayIterForTest(float sparsity, size_t rows = kRows,
                                size_t cols = kCols, size_t batches = kBatches);
  int Next() override;
  ~CudaArrayIterForTest() override = default;
};

inline void Reset(DataIterHandle self) {
  static_cast<ArrayIterForTest*>(self)->Reset();
}

inline int Next(DataIterHandle self) {
  return static_cast<ArrayIterForTest*>(self)->Next();
}
}  // namespace xgboost
#endif  // XGBOOST_TESTS_CPP_DATA_ITERATOR_FOR_TESTS_H_
