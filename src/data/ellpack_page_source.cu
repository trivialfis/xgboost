/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t, uint64_t, uint32_t
#include <memory>     // for shared_ptr, make_unique
#include <numeric>    // for accumulate
#include <tuple>      // for tie
#include <utility>    // for move

#include "../common/common.h"               // for HumanMemUnit, safe_cuda
#include "../common/cuda_context.cuh"       // for CUDAContext
#include "../common/cuda_rt_utils.h"        // for SetDevice
#include "../common/device_helpers.cuh"     // for CurrentDevice
#include "../common/numa_topo.h"            // for NumaMemCanCross, GetNumaMemBind
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "batch_utils.h"                    // for HostRatioIsAuto
#include "ellpack_page.cuh"                 // for EllpackPageImpl
#include "ellpack_page.h"                   // for EllpackPage
#include "ellpack_page_source.h"
#include "proxy_dmatrix.cuh"  // for DispatchAny
#include "xgboost/base.h"     // for bst_idx_t

namespace xgboost::data {
/**
 * Cache
 */
EllpackCache::EllpackCache(EllpackCacheInfo cinfo, StringView name)
    : file_name{name},
      cache_mapping{std::move(cinfo.cache_mapping)},
      buffer_bytes{std::move(cinfo.buffer_bytes)},
      buffer_rows{std::move(cinfo.buffer_rows)},
      cache_host_ratio{cinfo.cache_host_ratio} {
  CHECK_EQ(buffer_bytes.size(), buffer_rows.size());
  CHECK(!detail::HostRatioIsAuto(this->cache_host_ratio));
  CHECK_GE(this->cache_host_ratio, 0.0) << error::CacheHostRatioInvalid();
  CHECK_LE(this->cache_host_ratio, 1.0) << error::CacheHostRatioInvalid();
  if (!this->OnHost()) {
    file = std::make_unique<common::CuFileStream>(file_name, true);
  }
}

EllpackCache::~EllpackCache() = default;

std::size_t EllpackCache::SizeBytes() const {
  auto it = common::MakeIndexTransformIter([&](auto i) { return this->SizeBytes(i); });
  return std::accumulate(it, it + this->Size(), std::size_t{0});
}

std::size_t EllpackCache::DeviceSizeBytes() const {
  auto it = common::MakeIndexTransformIter([&](auto i) { return d_pages.at(i).size_bytes(); });
  return std::accumulate(it, it + this->Size(), std::size_t{0});
}

std::size_t EllpackCache::SizeBytes(std::size_t i) const {
  return pages.at(i)->MemCostBytes() + d_pages.at(i).size_bytes() +
         (this->OnHost() ? 0 : this->ExternalSizeBytes(i));
}

std::size_t EllpackCache::ExternalSizeBytes(std::size_t i) const {
  return this->OnHost() ? pages.at(i)->gidx_buffer.size_bytes()
                        : file_offsets.at(i + 1) - file_offsets.at(i);
}

std::size_t EllpackCache::GidxSizeBytes(std::size_t i) const {
  return this->ExternalSizeBytes(i) + d_pages.at(i).size_bytes();
}

std::size_t EllpackCache::GidxSizeBytes() const {
  auto it = common::MakeIndexTransformIter([&](auto i) { return this->GidxSizeBytes(i); });
  return std::accumulate(it, it + this->Size(), std::size_t{0});
}

class EllpackCacheStreamImpl {
  std::shared_ptr<EllpackCache> cache_;
  std::size_t ptr_{0};

  void Store(Context const* ctx, EllpackPageImpl const* src) {
    auto& cache = *cache_;
    CHECK_EQ(src->gidx_buffer.Resource()->Type(), common::ResourceHandler::kCudaMalloc);
    auto stream = ctx->CUDACtx()->Stream();
    auto size = src->gidx_buffer.size_bytes();
    auto ratio = cache.cache_host_ratio;
    auto external = ratio == 1.0 ? size
                    : ratio == 0.0
                        ? 0
                        : std::max(static_cast<std::size_t>(size * ratio), std::size_t{1});
    auto stored = std::make_unique<EllpackPageImpl>();
    stored->CopyInfo(src);

    auto device = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(size - external);
    if (!device.empty()) {
      dh::safe_cuda(cudaMemcpyAsync(device.data(), src->gidx_buffer.data() + external,
                                    device.size_bytes(), cudaMemcpyDefault, stream));
    }
    if (cache.OnHost()) {
      stored->gidx_buffer = common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(external);
      if (external != 0) {
        dh::safe_cuda(cudaMemcpyAsync(stored->gidx_buffer.data(), src->gidx_buffer.data(), external,
                                      cudaMemcpyDefault, stream));
      }
    } else {
      cache.file->WriteAsync(src->gidx_buffer.data(), external, cache.file_offsets.back(), stream);
      cache.file->Sync();
      cache.file_offsets.push_back(cache.file_offsets.back() + external);
    }
    // Both transfers must finish before releasing the input or assembly buffer.
    stream.Sync();
    cache.pages.push_back(std::move(stored));
    cache.d_pages.push_back(std::move(device));
    LOG(INFO) << "Create cache page with size:"
              << common::HumanMemUnit(cache.SizeBytes(cache.Size() - 1));
  }

 public:
  explicit EllpackCacheStreamImpl(std::shared_ptr<EllpackCache> cache) : cache_{std::move(cache)} {}

  auto Share() const { return cache_; }

  void Seek(bst_idx_t offset_bytes) {
    std::size_t n_bytes{0};
    ptr_ = 0;
    while (ptr_ < cache_->Size() && n_bytes < offset_bytes) {
      n_bytes += cache_->SizeBytes(ptr_++);
    }
    CHECK_EQ(n_bytes, offset_bytes) << "Invalid cache offset.";
  }

  [[nodiscard]] bool Write(Context const* ctx, EllpackPage const& page) {
    auto& cache = *cache_;
    CHECK_LT(cache.input_idx, cache.cache_mapping.size());
    auto group = cache.cache_mapping.at(cache.input_idx++);
    CHECK_EQ(group, cache.Size());
    bool last = cache.input_idx == cache.cache_mapping.size();
    bool complete = last || cache.cache_mapping.at(cache.input_idx) != group;
    auto src = page.Impl();

    if (!cache.NoConcat()) {
      if (!cache.pending) {
        cache.pending = std::make_unique<EllpackPageImpl>();
        cache.pending->CopyInfo(src);
        cache.pending->SetCuts(src->CutsShared());
        cache.pending->n_rows = cache.buffer_rows.at(group);
        cache.pending->gidx_buffer = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(
            ctx, cache.buffer_bytes.at(group), 0);
        cache.pending_offset = 0;
      }
      cache.pending_offset += cache.pending->Copy(ctx, src, cache.pending_offset);
      src = cache.pending.get();
    }
    if (!complete) {
      return false;
    }
    if (cache.pending) {
      CHECK_EQ(cache.pending_offset, src->n_rows * src->info.row_stride);
    }
    this->Store(ctx, src);
    cache.pending.reset();

    return true;
  }

  void Read(Context const* ctx, EllpackPage* out, bool prefetch_copy) const {
    auto const& cache = *cache_;
    auto const* stored = cache.pages.at(ptr_).get();
    auto const& device = cache.d_pages.at(ptr_);
    auto external = cache.ExternalSizeBytes(ptr_);
    auto dst = out->Impl();
    auto stream = ctx->CUDACtx()->Stream();

    if (external == 0) {
      // Fully resident pages need no transfer for either backend.
      dst->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          device.Resource()->DataAs<common::CompressedByteT>(), device.size(), device.Resource()};
    } else if (prefetch_copy || !cache.OnHost()) {
      dst->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(cache.GidxSizeBytes(ptr_));
      if (cache.OnHost()) {
        dh::safe_cuda(cudaMemcpyAsync(dst->gidx_buffer.data(), stored->gidx_buffer.data(), external,
                                      cudaMemcpyDefault, stream));
      } else {
        common::CuFileStream reader{*cache.file};
        reader.ReadAsync(dst->gidx_buffer.data(), external, cache.file_offsets.at(ptr_), stream);
        reader.Sync();
      }
      if (!device.empty()) {
        dh::safe_cuda(cudaMemcpyAsync(dst->gidx_buffer.data() + external, device.data(),
                                      device.size_bytes(), cudaMemcpyDefault, stream));
      }
    } else {
      // HMM/ATS can expose host memory directly; file storage always requires a read.
      auto const& host = stored->gidx_buffer;
      dst->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          host.Resource()->DataAs<common::CompressedByteT>(), host.size(), host.Resource()};
      if (!device.empty()) {
        dst->d_gidx_buffer = common::RefResourceView<common::CompressedByteT const>{
            device.data(), device.size(), device.Resource()};
      }
    }
    dst->CopyInfo(stored);
  }
};

/**
 * EllpackCacheStream
 */
EllpackCacheStream::EllpackCacheStream(std::shared_ptr<EllpackCache> cache)
    : p_impl_{std::make_unique<EllpackCacheStreamImpl>(std::move(cache))} {}

EllpackCacheStream::~EllpackCacheStream() = default;

std::shared_ptr<EllpackCache const> EllpackCacheStream::Share() const { return p_impl_->Share(); }

void EllpackCacheStream::Seek(bst_idx_t offset_bytes) { this->p_impl_->Seek(offset_bytes); }

void EllpackCacheStream::Read(Context const* ctx, EllpackPage* page, bool prefetch_copy) const {
  this->p_impl_->Read(ctx, page, prefetch_copy);
}

[[nodiscard]] bool EllpackCacheStream::Write(Context const* ctx, EllpackPage const& page) {
  return this->p_impl_->Write(ctx, page);
}

/**
 * EllpackFormatPolicy
 */
template <typename S>
void EllpackFormatPolicy<S>::DestroyPage(std::shared_ptr<S>* page) const {
  if (page && ctx_) {
    ctx_->CUDACtx()->Stream().Sync();
  }
  page->reset();
}

template void EllpackFormatPolicy<EllpackPage>::DestroyPage(
    std::shared_ptr<EllpackPage>* page) const;

/**
 * EllpackCacheStreamPolicy
 */
template <typename S, template <typename> typename F, bool on_host>
std::unique_ptr<typename EllpackCacheStreamPolicy<S, F, on_host>::WriterT>
EllpackCacheStreamPolicy<S, F, on_host>::CreateWriter(StringView name, std::uint32_t iter) {
  if (!p_cache_) {
    if constexpr (!on_host) {
      CHECK(!name.empty());
    }
    p_cache_ = std::make_shared<EllpackCache>(this->CacheInfo(), on_host ? StringView{} : name);
  }
  CHECK_EQ(iter, p_cache_->input_idx);
  return std::make_unique<WriterT>(p_cache_);
}

template <typename S, template <typename> typename F, bool on_host>
std::unique_ptr<typename EllpackCacheStreamPolicy<S, F, on_host>::ReaderT>
EllpackCacheStreamPolicy<S, F, on_host>::CreateReader(StringView, bst_idx_t offset,
                                                      bst_idx_t) const {
  auto fi = std::make_unique<ReaderT>(p_cache_);
  fi->Seek(offset);
  return fi;
}

template class EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy, true>;
template class EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy, false>;

void CalcCacheMapping(Context const* ctx, bool is_dense,
                      std::shared_ptr<common::HistogramCuts const> cuts,
                      std::int64_t min_cache_page_bytes, ExternalDataInfo const& ext_info,
                      bool is_validation, EllpackCacheInfo* cinfo) {
  CHECK(cinfo->param.Initialized()) << "Need to initialize scalar fields first.";
  auto ell_info = CalcNumSymbols(ctx, ext_info.row_stride, is_dense, cuts);

  /**
   * Configure the cache
   */
  // The total size of the cache.
  std::size_t n_cache_bytes = 0;
  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rowids.at(i + 1) - ext_info.base_rowids[i];
    auto n_bytes = common::CompressedBufferWriter::CalculateBufferSize(
        ext_info.row_stride * n_samples, ell_info.n_symbols);
    n_cache_bytes += n_bytes;
  }
  std::tie(cinfo->cache_host_ratio, min_cache_page_bytes) = detail::DftPageSizeHostRatio(
      n_cache_bytes, is_validation, cinfo->cache_host_ratio, min_cache_page_bytes);

  /**
   * Calculate the cache buffer size
   */
  std::vector<std::size_t> cache_bytes;
  std::vector<std::size_t> cache_mapping(ext_info.n_batches, 0);
  std::vector<std::size_t> cache_rows;

  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rowids[i + 1] - ext_info.base_rowids[i];
    auto n_bytes = common::CompressedBufferWriter::CalculateBufferSize(
        ext_info.row_stride * n_samples, ell_info.n_symbols);

    if (cache_bytes.empty()) {
      // Push the first page
      cache_bytes.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    } else if (static_cast<decltype(min_cache_page_bytes)>(cache_bytes.back()) <
               min_cache_page_bytes) {
      // Concatenate to the previous page
      cache_bytes.back() += n_bytes;
      cache_rows.back() += n_samples;
    } else {
      // Push a new page
      cache_bytes.push_back(n_bytes);
      cache_rows.push_back(n_samples);
    }
    cache_mapping[i] = cache_bytes.size() - 1;
  }

  cinfo->cache_mapping = std::move(cache_mapping);
  cinfo->buffer_bytes = std::move(cache_bytes);
  cinfo->buffer_rows = std::move(cache_rows);

  // Directly store in device if there's only one batch.
  if (cinfo->NumBatchesCc() == 1) {
    cinfo->cache_host_ratio = 0.0;
  }

  LOG(INFO) << "`cache_host_ratio`=" << cinfo->cache_host_ratio
            << " `min_cache_page_bytes`=" << min_cache_page_bytes;
}

/**
 * EllpackPageSourceImpl
 */
template <typename F>
void EllpackPageSourceImpl<F>::Fetch() {
  curt::SetDevice(this->Device().ordinal);
  if (!this->ReadCache()) {
    if (this->Iter() != 0 && !this->sync_) {
      // source is initialized to be the 0th page during construction, so when count_ is 0
      // there's no need to increment the source.
      ++(*this->source_);
    }
    // This is not read from cache so we still need it to be synced with sparse page source.
    CHECK_EQ(this->Iter(), this->source_->Iter());
    auto const& csr = this->source_->Page();
    this->DestroyPage(&this->page_);
    this->page_.reset(new EllpackPage{});
    auto* impl = this->page_->Impl();
    if (this->GetCuts()->HasCategorical()) {
      CHECK(!this->feature_types_.empty());
    }
    *impl =
        EllpackPageImpl{this->Ctx(), this->GetCuts(), *csr, is_dense_, row_stride_, feature_types_};
    this->page_->SetBaseRowId(csr->base_rowid);
    LOG(INFO) << "Generated an Ellpack page with size: "
              << common::HumanMemUnit(impl->MemCostBytes())
              << " from a SparsePage with size:" << common::HumanMemUnit(csr->MemCostBytes());
    this->WriteCache();
  }
}

// Instantiation
template void
EllpackPageSourceImpl<DefaultFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
EllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
EllpackPageSourceImpl<EllpackFileStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();

/**
 * ExtEllpackPageSourceImpl
 */
template <typename F>
void ExtEllpackPageSourceImpl<F>::Fetch() {
  curt::SetDevice(this->Device().ordinal);
  if (!this->ReadCache()) {
    auto iter = this->source_->Iter();
    CHECK_EQ(this->Iter(), iter);
    cuda_impl::DispatchAny(proxy_, [this](auto const& value) {
      CHECK(this->proxy_->Ctx()->IsCUDA()) << "All batches must use the same device type.";
      proxy_->Info().feature_types.SetDevice(dh::GetDevice(this->ctx_));
      auto d_feature_types = proxy_->Info().feature_types.ConstDeviceSpan();
      auto n_samples = value.NumRows();
      if (this->GetCuts()->HasCategorical()) {
        CHECK(!d_feature_types.empty());
      }
      dh::device_vector<size_t> row_counts(n_samples + 1, 0);
      common::Span<size_t> row_counts_span(row_counts.data().get(), row_counts.size());
      bst_idx_t row_stride = GetRowCounts(this->ctx_, value, row_counts_span,
                                          dh::GetDevice(this->ctx_), this->missing_);
      CHECK_LE(row_stride, this->ext_info_.row_stride);
      this->DestroyPage(&this->page_);
      this->page_.reset(new EllpackPage{});
      *this->page_->Impl() = EllpackPageImpl{this->ctx_,
                                             value,
                                             this->missing_,
                                             this->info_->IsDense(),
                                             row_counts_span,
                                             d_feature_types,
                                             this->ext_info_.row_stride,
                                             n_samples,
                                             this->GetCuts()};
      this->info_->Extend(proxy_->Info(), false, true);
    });
    LOG(DEBUG) << "Generated an Ellpack page with size: "
               << common::HumanMemUnit(this->page_->Impl()->MemCostBytes())
               << " from an batch with estimated size: "
               << cuda_impl::DispatchAny<false>(proxy_, [](auto const& adapter) {
                    return common::HumanMemUnit(adapter->SizeBytes());
                  });
    this->page_->SetBaseRowId(this->ext_info_.base_rowids.at(iter));
    this->WriteCache();
  }
}

// Instantiation
template void
ExtEllpackPageSourceImpl<DefaultFormatStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
ExtEllpackPageSourceImpl<EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
template void
ExtEllpackPageSourceImpl<EllpackFileStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();

namespace detail {
void EllpackFormatCheckNuma(StringView msg) {
#if defined(__linux__)
  bool can_cross = common::NumaMemCanCross();
  std::uint32_t numa = 0;
  auto incorrect = [&numa] {
    std::uint32_t cpu = 0;
    return common::GetCpuNuma(&cpu, &numa) && static_cast<std::int32_t>(numa) != curt::GetNumaId();
  };

  if (can_cross && !common::GetNumaMemBind()) {
    LOG(WARNING) << "Running on a NUMA system without membind." << msg;
  } else if (can_cross && incorrect()) {
    LOG(WARNING) << "Incorrect NUMA CPU bind, CPU node:" << numa
                 << ", GPU node:" << curt::GetNumaId() << "." << msg;
  }
#else
  (void)msg;
#endif
}
}  // namespace detail
}  // namespace xgboost::data
