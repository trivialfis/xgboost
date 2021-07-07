/*!
 * Copyright 2020-2021 XGBoost contributors
 */
#include <xgboost/c_api.h>
#include "data_iterator_for_test.h"
#include "../../src/data/iterative_device_dmatrix.h"

namespace xgboost {

ArrayIterForTest::ArrayIterForTest(float sparsity, size_t rows, size_t cols,
                                   size_t batches) : rows_{rows}, cols_{cols}, n_batches_{batches} {
  XGProxyDMatrixCreate(&proxy_);
  rng_.reset(new RandomDataGenerator{rows_, cols_, sparsity});
  std::tie(batches_, interface_) =
      rng_->GenerateArrayInterfaceBatch(&data_, n_batches_);
}

ArrayIterForTest::~ArrayIterForTest() { XGDMatrixFree(proxy_); }

int ArrayIterForTest::Next() {
  if (iter_ == n_batches_) {
    return 0;
  }
  XGProxyDMatrixSetDataDense(proxy_, batches_[iter_].c_str());
  iter_++;
  return 1;
}

size_t constexpr ArrayIterForTest::kRows;
size_t constexpr ArrayIterForTest::kCols;

CudaArrayIterForTest::CudaArrayIterForTest(float sparsity, size_t rows,
                                           size_t cols, size_t batches)
    : ArrayIterForTest{sparsity, rows, cols, batches} {
  rng_->Device(0);
  std::tie(batches_, interface_) =
      rng_->GenerateArrayInterfaceBatch(&data_, n_batches_);
  this->Reset();
}

size_t constexpr CudaArrayIterForTest::kRows;
size_t constexpr CudaArrayIterForTest::kCols;
size_t constexpr CudaArrayIterForTest::kBatches;

int CudaArrayIterForTest::Next() {
  if (iter_ == n_batches_) {
    return 0;
  }
  XGProxyDMatrixSetDataCudaArrayInterface(proxy_, batches_[iter_].c_str());
  iter_++;
  return 1;
}
}  // namespace xgboost
