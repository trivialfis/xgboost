/*!
 * Copyright 2015 by Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>

#include <memory>
#include <string>


namespace xgboost {
namespace data {
// Used for single batch data.
class SimpleDMatrix : public DMatrix {
 public:
  SimpleDMatrix() = default;
  template <typename AdapterT>
  explicit SimpleDMatrix(AdapterT* adapter, float missing, int nthread);

  explicit SimpleDMatrix(dmlc::Stream* in_stream);
  ~SimpleDMatrix() override = default;

  void SaveToLocalFile(const std::string& fname);

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  bool SingleColBlock() const override { return true; }
  DMatrix* Slice(Context const* ctx, common::Span<int32_t const> ridxs) override;

  /*! \brief magic number used to identify SimpleDMatrix binary files */
  static const int kMagic = 0xffffab01;

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches(GenericParameter const* ctx) override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches(GenericParameter const* ctx) override;
  BatchSet<EllpackPage> GetEllpackBatches(GenericParameter const *ctx,
                                          const BatchParam &param) override;

  MetaInfo info_;
  SparsePage sparse_page_;  // Primary storage type
  std::unique_ptr<CSCPage> column_page_;
  std::unique_ptr<SortedCSCPage> sorted_column_page_;
  std::unique_ptr<EllpackPage> ellpack_page_;
  BatchParam batch_param_;

  bool EllpackExists() const override {
    return static_cast<bool>(ellpack_page_);
  }
  bool SparsePageExists() const override {
    return true;
  }
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
