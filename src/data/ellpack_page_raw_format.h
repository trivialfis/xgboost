/**
 * Copyright 2019-2026, XGBoost contributors
 */
#pragma once

#include <array>    // for array
#include <cstddef>  // for size_t
#include <memory>   // for shared_ptr
#include <mutex>    // for mutex
#include <utility>  // for move

#include "../common/compressed_iterator.h"  // for CompressedByteT
#include "../common/io.h"                   // for AlignedResourceReadStream
#include "../common/ref_resource_view.h"    // for RefResourceView
#include "batch_utils.h"                   // for DftPrefetchBatches
#include "sparse_page_writer.h"             // for SparsePageFormat
#include "xgboost/data.h"                   // for EllpackPage

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif                         // !defined(XGBOOST_USE_CUDA)`

namespace xgboost::common {
class HistogramCuts;
}

namespace xgboost::data {

struct Cache;
class EllpackHostCacheStream;

// Device buffers matching GPU prefetching. Views keep a buffer leased until the
// page and any other references to its storage have been released. Device work must
// finish before releasing a view (see EllpackFormatPolicy::DestroyPage).
class EllpackPagePool {
  std::array<std::shared_ptr<common::ResourceHandler>, ::xgboost::cuda_impl::DftPrefetchBatches()>
      buffers_;
  std::size_t max_page_bytes_;
  std::mutex mutex_;

 public:
  explicit EllpackPagePool(std::size_t max_page_bytes) : max_page_bytes_{max_page_bytes} {}
  explicit EllpackPagePool(Cache const& cache);

  [[nodiscard]] common::RefResourceView<common::CompressedByteT> Allocate(std::size_t n_bytes);
};

class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
  std::shared_ptr<common::HistogramCuts const> cuts_;
  DeviceOrd device_;
  BatchParam param_;
  // Supports CUDA HMM or ATS
  bool has_hmm_ats_{false};
  Context const* ctx_;
  // Required for reads; absent while writing the cache.
  EllpackPagePool* pool_;

 public:
  explicit EllpackPageRawFormat(Context const* ctx,
                                std::shared_ptr<common::HistogramCuts const> cuts, DeviceOrd device,
                                BatchParam param, bool has_hmm_ats, EllpackPagePool* pool)
      : cuts_{std::move(cuts)},
        device_{device},
        param_{std::move(param)},
        has_hmm_ats_{has_hmm_ats},
        ctx_{ctx},
        pool_{pool} {}
  [[nodiscard]] bool Read(EllpackPage* page, common::AlignedResourceReadStream* fi) override;
  [[nodiscard]] std::size_t Write(EllpackPage const& page,
                                  common::AlignedFileWriteStream* fo) override;

  [[nodiscard]] bool Read(EllpackPage* page, EllpackHostCacheStream* fi) const;
  [[nodiscard]] std::size_t Write(EllpackPage const& page, EllpackHostCacheStream* fo) const;
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
