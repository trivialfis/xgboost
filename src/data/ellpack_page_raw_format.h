/**
 * Copyright 2019-2024, XGBoost contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
#include <utility>  // for move

#include "../common/cuda_rt_utils.h"  // for CudaPrefetchConfig
#include "../common/io.h"             // for AlignedResourceReadStream
#include "sparse_page_writer.h"       // for SparsePageFormat
#include "xgboost/data.h"             // for EllpackPage

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif                         // !defined(XGBOOST_USE_CUDA)`

namespace xgboost::common {
class HistogramCuts;
}

namespace xgboost::data {

class EllpackHostCacheStream;

class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
  std::shared_ptr<common::HistogramCuts const> cuts_;
  DeviceOrd device_;
  // Supports CUDA HMM or ATS
  bool has_hmm_ats_{false};
  common::CudaPrefetchConfig config_;

 public:
  explicit EllpackPageRawFormat(std::shared_ptr<common::HistogramCuts const> cuts, DeviceOrd device,
                                bool has_hmm_ats, common::CudaPrefetchConfig const& config)
      : cuts_{std::move(cuts)}, device_{device}, has_hmm_ats_{has_hmm_ats}, config_{config} {}
  [[nodiscard]] bool Read(EllpackPage* page, common::AlignedResourceReadStream* fi) override;
  [[nodiscard]] std::size_t Write(const EllpackPage& page,
                                  common::AlignedFileWriteStream* fo) override;

  [[nodiscard]] bool Read(EllpackPage* page, EllpackHostCacheStream* fi) const;
  [[nodiscard]] std::size_t Write(const EllpackPage& page, EllpackHostCacheStream* fo) const;
};

#if !defined(XGBOOST_USE_CUDA)
inline bool EllpackPageRawFormat::Read(EllpackPage*, common::AlignedResourceReadStream*) {
  common::AssertGPUSupport();
  return false;
}

inline std::size_t EllpackPageRawFormat::Write(const EllpackPage&,
                                               common::AlignedFileWriteStream*) {
  common::AssertGPUSupport();
  return 0;
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data
