/**
 * Copyright 2019-2026, XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <cstdint>  // for int32_t
#include <limits>   // for numeric_limits
#include <memory>   // for shared_ptr
#include <string>   // for string
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/compressed_iterator.h"  // for CompressedByteT
#include "../common/cuda_rt_utils.h"        // for SupportsPageableMem, SupportsAts
#include "../common/cufile_stream.h"        // for CuFileStream, InitCuFile
#include "../common/hist_util.h"            // for HistogramCuts
#include "../common/ref_resource_view.h"    // for RefResourceView
#include "../data/batch_utils.h"            // for AutoHostRatio
#include "ellpack_page.h"                   // for EllpackPage
#include "ellpack_page_raw_format.h"        // for EllpackPageRawFormat
#include "sparse_page_source.h"             // for PageSourceIncMixIn
#include "xgboost/base.h"                   // for bst_idx_t
#include "xgboost/context.h"                // for DeviceOrd
#include "xgboost/data.h"                   // for BatchParam
#include "xgboost/span.h"                   // for Span

namespace xgboost::data {
struct EllpackCacheInfo {
  BatchParam param;
  // The fraction of the cache stored externally (host memory or file).
  double cache_host_ratio{::xgboost::cuda_impl::AutoHostRatio()};
  float missing{std::numeric_limits<float>::quiet_NaN()};
  std::vector<bst_idx_t> cache_mapping;
  std::vector<bst_idx_t> buffer_bytes;  // N bytes of the concatenated pages.
  std::vector<bst_idx_t> buffer_rows;   // The number of rows for each batch after concatenation.

  EllpackCacheInfo() = default;
  EllpackCacheInfo(BatchParam param, double h_ratio, float missing)
      : param{std::move(param)}, cache_host_ratio{h_ratio}, missing{missing} {}
  EllpackCacheInfo(BatchParam param, ExtMemConfig const& config)
      : param{std::move(param)},
        cache_host_ratio{config.cache_host_ratio},
        missing{config.missing} {}

  // The number of batches for the concatenated cache.
  [[nodiscard]] std::size_t NumBatchesCc() const { return this->buffer_rows.size(); }
};

// Storage is shared; each prefetch reader has an independent cursor/request.
// Page metadata and the device suffix are identical for host and file backends.
struct EllpackCache {
  // Metadata and, for the host backend, the external prefix.
  std::vector<std::unique_ptr<EllpackPageImpl>> pages;
  using DPage = common::RefResourceView<common::CompressedByteT>;
  std::vector<DPage> d_pages;

  std::string const file_name;
  // Physical file offsets, separate from logical cache page sizes.
  std::vector<bst_idx_t> file_offsets{0};
  // Keep the registration and IO stream alive while prefetch readers share them.
  std::unique_ptr<common::CuFileStream> file;

  // Only one concatenated page is under construction at a time.
  std::unique_ptr<EllpackPageImpl> pending;
  bst_idx_t input_idx{0};
  bst_idx_t pending_offset{0};
  std::vector<bst_idx_t> const cache_mapping;
  std::vector<bst_idx_t> const buffer_bytes;
  std::vector<bst_idx_t> const buffer_rows;
  double const cache_host_ratio;

  explicit EllpackCache(EllpackCacheInfo cinfo, StringView file_name = {});
  ~EllpackCache();

  [[nodiscard]] bool OnHost() const { return file_name.empty(); }
  // Logical sizes include metadata and both external and device payloads.
  [[nodiscard]] std::size_t SizeBytes() const;
  [[nodiscard]] std::size_t SizeBytes(std::size_t i) const;
  [[nodiscard]] std::size_t DeviceSizeBytes() const;
  [[nodiscard]] std::size_t ExternalSizeBytes(std::size_t i) const;
  [[nodiscard]] std::size_t GidxSizeBytes(std::size_t i) const;
  [[nodiscard]] std::size_t GidxSizeBytes() const;
  [[nodiscard]] std::size_t Size() const { return this->pages.size(); }
  [[nodiscard]] bool NoConcat() const { return cache_mapping.size() == buffer_rows.size(); }
};

// Pimpl to hide CUDA calls from the host compiler.
class EllpackCacheStreamImpl;

/**
 * @brief A view of shared ELLPACK cache storage.
 */
class EllpackCacheStream {
  std::unique_ptr<EllpackCacheStreamImpl> p_impl_;

 public:
  explicit EllpackCacheStream(std::shared_ptr<EllpackCache> cache);
  ~EllpackCacheStream();
  /**
   * @brief Get a shared handler to the cache.
   */
  std::shared_ptr<EllpackCache const> Share() const;
  /**
   * @brief Stream seek.
   *
   * @param offset_bytes This must align to the actual cached page size.
   */
  void Seek(bst_idx_t offset_bytes);
  /**
   * @brief Read a page from the cache.
   *
   * The read page might be concatenated during page write.
   *
   * @param page[out] The returned page.
   * @param prefetch_copy[in] Does the stream need to copy the page?
   */
  void Read(Context const* ctx, EllpackPage* page, bool prefetch_copy) const;
  /**
   * @brief Append an input page and store the cache page when its group is complete.
   *
   * Inputs in the same group are concatenated before storing the completed page.
   *
   * @return Whether a completed cache page was stored.
   */
  [[nodiscard]] bool Write(Context const* ctx, EllpackPage const& page);
};

namespace detail {
// Not a member of `EllpackFormatPolicy`. Hide the impl without requiring template specialization.
void EllpackFormatCheckNuma(StringView msg);
}  // namespace detail

template <typename S>
class EllpackFormatPolicy {
  std::shared_ptr<common::HistogramCuts const> cuts_{nullptr};
  DeviceOrd device_;
  bool has_hmm_{curt::SupportsPageableMem()};
  Context const* ctx_{nullptr};

  EllpackCacheInfo cache_info_;
  static_assert(std::is_same_v<S, EllpackPage>);

 public:
  using FormatT = EllpackPageRawFormat;

 public:
  EllpackFormatPolicy() {
    StringView msg{" The overhead of iterating through external memory might be significant."};
    if (!(has_hmm_ || curt::SupportsAts())) {
      LOG(WARNING) << "CUDA heterogeneous memory management is not available." << msg;
    }
    if (!(GlobalConfigThreadLocalStore::Get()->use_rmm ||
          GlobalConfigThreadLocalStore::Get()->use_cuda_async_pool)) {
      LOG(WARNING) << "Neither `use_rmm` nor `use_cuda_async_pool` is enabled." << msg;
    }
    if (GlobalConfigThreadLocalStore::Get()->use_rmm) {
#if !defined(XGBOOST_USE_RMM)
      LOG(WARNING) << "XGBoost is not built with RMM support. But the `use_rmm` flag is enabled.";
#endif
    }
    std::int32_t major{0}, minor{0};
    curt::GetDrVersionGlobal(&major, &minor);
    if ((major < 12 || (major == 12 && minor < 7)) && curt::SupportsAts()) {
      // Use ATS, but with an old kernel driver.
      LOG(WARNING) << "Using an old kernel driver with supported CTK<12.7."
                   << "The latest version of CTK supported by the current driver: " << major << "."
                   << minor << "." << msg;
    }
    detail::EllpackFormatCheckNuma(msg);
  }
  // For testing with the HMM flag.
  explicit EllpackFormatPolicy(bool has_hmm) : has_hmm_{has_hmm} {}

  [[nodiscard]] auto CreatePageFormat(BatchParam const& param) const {
    CHECK_EQ(cuts_->cut_values_.Device(), device_);
    std::unique_ptr<FormatT> fmt{new EllpackPageRawFormat{ctx_, cuts_, device_, param, has_hmm_}};
    return fmt;
  }
  void SetCuts(Context const* ctx, std::shared_ptr<common::HistogramCuts const> cuts,
               DeviceOrd device, EllpackCacheInfo cinfo) {
    this->ctx_ = ctx;
    std::swap(this->cuts_, cuts);
    this->device_ = device;
    CHECK(this->device_.IsCUDA());
    this->cache_info_ = std::move(cinfo);
  }
  [[nodiscard]] auto GetCuts() const {
    CHECK(cuts_);
    return cuts_;
  }
  [[nodiscard]] auto Device() const { return this->device_; }
  [[nodiscard]] auto const& CacheInfo() { return this->cache_info_; }
  [[nodiscard]] auto Ctx() const { return this->ctx_; }
  void DestroyPage(std::shared_ptr<S>* page) const;
};

template <typename S, template <typename> typename F, bool on_host = true>
class EllpackCacheStreamPolicy : public F<S> {
  std::shared_ptr<EllpackCache> p_cache_;

 public:
  using WriterT = EllpackCacheStream;
  using ReaderT = EllpackCacheStream;

  EllpackCacheStreamPolicy() {
    if constexpr (!on_host) {
      common::InitCuFile();
    }
  }
  // For testing with the HMM flag.
  explicit EllpackCacheStreamPolicy(bool has_hmm) : F<S>{has_hmm} {
    if constexpr (!on_host) {
      common::InitCuFile();
    }
  }
  [[nodiscard]] std::unique_ptr<WriterT> CreateWriter(StringView name, std::uint32_t iter);
  [[nodiscard]] std::unique_ptr<ReaderT> CreateReader(StringView name, bst_idx_t offset,
                                                      bst_idx_t length) const;
  std::shared_ptr<EllpackCache const> Share() const { return p_cache_; }
};

template <typename S, template <typename> typename F>
using EllpackFileStreamPolicy = EllpackCacheStreamPolicy<S, F, false>;

/**
 * @brief Calculate the size of each internal cached page along with the mapping of old
 *        pages to the new pages.
 */
void CalcCacheMapping(Context const* ctx, bool is_dense,
                      std::shared_ptr<common::HistogramCuts const> cuts,
                      std::int64_t min_cache_page_bytes, ExternalDataInfo const& ext_info,
                      bool is_validation, EllpackCacheInfo* cinfo);

/**
 * @brief Ellpack source with sparse pages as the underlying source.
 */
template <typename F>
class EllpackPageSourceImpl : public PageSourceIncMixIn<EllpackPage, F> {
  using Super = PageSourceIncMixIn<EllpackPage, F>;
  bool is_dense_;
  bst_idx_t row_stride_;
  BatchParam param_;
  common::Span<FeatureType const> feature_types_;

 public:
  EllpackPageSourceImpl(Context const* ctx, bst_feature_t n_features, std::size_t n_batches,
                        std::shared_ptr<Cache> cache, std::shared_ptr<common::HistogramCuts> cuts,
                        bool is_dense, bst_idx_t row_stride,
                        common::Span<FeatureType const> feature_types,
                        std::shared_ptr<SparsePageSource> source, EllpackCacheInfo const& cinfo)
      : Super{cinfo.missing, ctx->Threads(), n_features, n_batches, cache, false},
        is_dense_{is_dense},
        row_stride_{row_stride},
        param_{std::move(cinfo.param)},
        feature_types_{feature_types} {
    this->source_ = source;
    cuts->SetDevice(ctx->Device());
    this->SetCuts(ctx, std::move(cuts), ctx->Device(), cinfo);
    this->Fetch();
  }

  void Fetch() final;
};

// Cache to host
using EllpackPageHostSource =
    EllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

// Cache to disk
using EllpackPageSource =
    EllpackPageSourceImpl<EllpackFileStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

/**
 * @brief Ellpack source directly interfaces with user-defined iterators.
 */
template <typename FormatCreatePolicy>
class ExtEllpackPageSourceImpl : public ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy> {
  using Super = ExtQantileSourceMixin<EllpackPage, FormatCreatePolicy>;

  Context const* ctx_;
  BatchParam p_;
  DMatrixProxy* proxy_;
  MetaInfo* info_;
  ExternalDataInfo ext_info_;

 public:
  ExtEllpackPageSourceImpl(
      Context const* ctx, MetaInfo* info, ExternalDataInfo ext_info, std::shared_ptr<Cache> cache,
      std::shared_ptr<common::HistogramCuts> cuts,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> source,
      DMatrixProxy* proxy, EllpackCacheInfo const& cinfo)
      : Super{cinfo.missing, ctx->Threads(), static_cast<bst_feature_t>(info->num_col_), source,
              cache},
        ctx_{ctx},
        p_{cinfo.param},
        proxy_{proxy},
        info_{info},
        ext_info_{std::move(ext_info)} {
    cuts->SetDevice(ctx->Device());
    this->SetCuts(ctx, std::move(cuts), ctx->Device(), cinfo);
    CHECK(!this->cache_info_->written);
    this->source_->Reset();
    CHECK(this->source_->Next());
    this->Fetch();
  }

  void Fetch() final;
  // Need a specialized end iter as we can concatenate pages.
  void EndIter() final {
    if (this->cache_info_->written) {
      CHECK_EQ(this->Iter(), this->cache_info_->Size());
    } else {
      CHECK_LE(this->cache_info_->Size(), this->ext_info_.n_batches);
    }
    this->cache_info_->Commit();
    CHECK_GE(this->count_, 1);
    this->count_ = 0;
  }
};

// Cache to host
using ExtEllpackPageHostSource =
    ExtEllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

// Cache to disk
using ExtEllpackPageSource =
    ExtEllpackPageSourceImpl<EllpackFileStreamPolicy<EllpackPage, EllpackFormatPolicy>>;

#if !defined(XGBOOST_USE_CUDA)
template <typename S>
inline void EllpackFormatPolicy<S>::DestroyPage(std::shared_ptr<S>* page) const {
  page->reset();
}

template <typename F>
inline void EllpackPageSourceImpl<F>::Fetch() {
  // silent the warning about unused variables.
  (void)(row_stride_);
  (void)(is_dense_);
  common::AssertGPUSupport();
}

template <typename F>
inline void ExtEllpackPageSourceImpl<F>::Fetch() {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::data

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
