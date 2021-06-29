/*!
 * Copyright 2020 by Contributors
 * \file iterative_device_dmatrix.h
 */
#ifndef XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/c_api.h"
#include "proxy_dmatrix.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {

struct Cache {
  bool written;
  std::string format;
  std::string id;
};

class IterativeDeviceDMatrix : public DMatrix {
  MetaInfo info_;
  BatchParam batch_param_;
  std::shared_ptr<EllpackPage> ellpack_page_;
  std::shared_ptr<SparsePage> sparse_page_;
  std::map<std::string, std::shared_ptr<Cache>> cache_info_;

  DMatrixHandle proxy_;
  DataIterHandle iter_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;
  float missing_;
  int nthreads_;
  size_t n_batches_ {0};

 public:
  void InitializeEllpack(DataIterHandle iter, float missing, int nthread);
  void InitializeExternalMemory(DataIterHandle iter, float missing, int nthread);

 public:
  explicit IterativeDeviceDMatrix(DataIterHandle iter, DMatrixHandle proxy,
                                  DataIterResetCallback *reset,
                                  XGDMatrixCallbackNext *next, float missing,
                                  int nthread, int max_bin)
      : proxy_{proxy}, iter_{iter}, reset_{reset}, next_{next},
        missing_{missing}, nthreads_{nthread} {
    batch_param_ = BatchParam{0, max_bin, 0};
    this->InitializeExternalMemory(iter_, missing_, nthreads_);
  }

  bool EllpackExists() const override { return true; }
  bool SparsePageExists() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const> ridxs) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Device DMatrix.";
    return nullptr;
  }

  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  bool SingleColBlock() const override { return false; }

  MetaInfo& Info() override {
    return info_;
  }
  MetaInfo const& Info() const override {
    return info_;
  }

  ~IterativeDeviceDMatrix() override {
    for (auto const& kv : cache_info_) {
      if (std::ifstream f{kv.first}) {
        TryDeleteCacheFile(kv.first);
      }
    }
  }
};
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_
