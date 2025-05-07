/**
 * Copyright 2019-2025, XGBoost contributors
 */
#include <algorithm>  // for count_if
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t, uint64_t, uint32_t
#include <memory>     // for shared_ptr, make_unique, make_shared
#include <numeric>    // for accumulate
#include <utility>    // for move

#include "../common/common.h"            // for HumanMemUnit, safe_cuda
#include "../common/cuda_rt_utils.h"     // for SetDevice
#include "../common/device_helpers.cuh"  // for CUDAStreamView, DefaultStream
#include "../common/nvcomp_format.h"
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/resource.cuh"           // for PrivateCudaMmapConstStream
#include "../common/transform_iterator.h"   // for MakeIndexTransformIter
#include "ellpack_page.cuh"                 // for EllpackPageImpl
#include "ellpack_page.h"                   // for EllpackPage
#include "ellpack_page_source.h"
#include "proxy_dmatrix.cuh"  // for Dispatch
#include "xgboost/base.h"     // for bst_idx_t

namespace xgboost::data {
namespace {
[[nodiscard]] bool IsDevicePage(EllpackPageImpl const* page) {
  switch (page->gidx_buffer.Resource()->Type()) {
    case common::ResourceHandler::kCudaMalloc:
    case common::ResourceHandler::kCudaGrowOnly: {
      return true;
    }
    case common::ResourceHandler::kCudaHostCache:
    case common::ResourceHandler::kCudaMmap:
    case common::ResourceHandler::kMmap:
    case common::ResourceHandler::kMalloc:
      return false;
  }
  LOG(FATAL) << "Unreachable";
  return false;
}
}  // anonymous namespace

/**
 * Cache
 */
EllpackMemCache::EllpackMemCache(EllpackCacheInfo cinfo)
    : cache_mapping{std::move(cinfo.cache_mapping)},
      buffer_bytes{std::move(cinfo.buffer_bytes)},
      buffer_rows{std::move(cinfo.buffer_rows)},
      cache_host_ratio{cinfo.cache_host_ratio},
      max_num_device_pages{cinfo.max_num_device_pages} {
  CHECK_EQ(buffer_bytes.size(), buffer_rows.size());
  CHECK_GT(cinfo.cache_host_ratio, 0.0);
}

EllpackMemCache::~EllpackMemCache() = default;

[[nodiscard]] std::size_t EllpackMemCache::SizeBytes() const {
  auto it = common::MakeIndexTransformIter([&](auto i) {
    return pages.at(i)->MemCostBytes() + this->d_pages.at(i).size_bytes() +
           this->c_pages.at(i).size_bytes();
  });
  using T = std::iterator_traits<decltype(it)>::value_type;
  return std::accumulate(it, it + pages.size(), static_cast<T>(0));
}

[[nodiscard]] EllpackPageImpl const* EllpackMemCache::At(std::int32_t k) const {
  return this->pages.at(k).get();
}

[[nodiscard]] std::int64_t EllpackMemCache::NumDevicePages() const {
  return std::count_if(this->pages.cbegin(), this->pages.cend(),
                       [](auto const& page) { return IsDevicePage(page.get()); });
}

/**
 * Cache stream.
 */
class EllpackHostCacheStreamImpl {
  std::shared_ptr<EllpackMemCache> cache_;
  std::int32_t ptr_{0};

 public:
  explicit EllpackHostCacheStreamImpl(std::shared_ptr<EllpackMemCache> cache)
      : cache_{std::move(cache)} {}

  auto Share() { return cache_; }

  void Seek(bst_idx_t offset_bytes) {
    std::size_t n_bytes{0};
    std::int32_t k{-1};
    for (std::size_t i = 0, n = cache_->pages.size(); i < n; ++i) {
      if (n_bytes == offset_bytes) {
        k = i;
        break;
      }
      n_bytes += (cache_->pages[i]->MemCostBytes() + cache_->d_pages[i].size_bytes());
    }
    if (offset_bytes == n_bytes && k == -1) {
      k = this->cache_->pages.size();  // seek end
    }
    CHECK_NE(k, -1) << "Invalid offset:" << offset_bytes;
    ptr_ = k;
  }

  [[nodiscard]] bool Write(EllpackPage const& page) {
    auto impl = page.Impl();
    auto ctx = Context{}.MakeCUDA(dh::CurrentDevice());

    this->cache_->sizes_orig.push_back(page.Impl()->MemCostBytes());
    auto orig_ptr = this->cache_->sizes_orig.size() - 1;

    CHECK_LT(orig_ptr, this->cache_->NumBatchesOrig());
    auto cache_idx = this->cache_->cache_mapping.at(orig_ptr);
    // Wrap up the previous page if this is a new page, or this is the last page.
    auto new_page = cache_idx == this->cache_->pages.size();
    // Last page expected from the user.
    auto last_page = (orig_ptr + 1) == this->cache_->NumBatchesOrig();

    bool const no_concat = this->cache_->NoConcat();
    // FIXME: no_concat, and to_device, estimate the ratio.

    // Whether the page should be cached in device. If true, then we don't need to make a
    // copy during write since the temporary page is already in device when page
    // concatenation is enabled.
    //
    // This applies only to a new cached page. If we are concatenating this page to an
    // existing cached page, then we should respect the existing flag obtained from the
    // first page of the cached page.
    auto cache_host_ratio = this->cache_->cache_host_ratio;
    CHECK_GT(cache_host_ratio, 0);
    auto get_host_nbytes = [&](EllpackPageImpl const* old_impl) {
      if (this->cache_->cache_host_ratio == 1.0) {
        return old_impl->gidx_buffer.size_bytes();
      }
      if (this->cache_->cache_host_ratio == 0.0) {
        return static_cast<std::size_t>(0);
      }
      auto n_bytes =
          std::max(static_cast<std::size_t>(old_impl->gidx_buffer.size_bytes() * cache_host_ratio),
                   std::size_t{1});
      return n_bytes;
    };
    auto commit_host_page = [cache_host_ratio, get_host_nbytes,
                             &ctx](EllpackPageImpl const* old_impl) {
      CHECK_EQ(old_impl->gidx_buffer.Resource()->Type(), common::ResourceHandler::kCudaMalloc);
      auto new_impl = std::make_unique<EllpackPageImpl>();
      new_impl->CopyInfo(old_impl);
      // Split the cache
      // The ratio of the page that will be on the host.

      // Host buffer
      auto n_bytes = get_host_nbytes(old_impl);
      // Further split into host buffer and compressed host buffer.
      CHECK_LE(n_bytes, old_impl->gidx_buffer.size_bytes());
      std::cout << "n_bytes:" << n_bytes << " sb:" << old_impl->gidx_buffer.size_bytes() << std::endl;
      auto n_compressed_bytes = n_bytes / 2;
      auto n_host_bytes = n_bytes - n_compressed_bytes;
      CHECK_GT(n_compressed_bytes, 0);
      CHECK_GT(n_host_bytes, 0);
      CHECK_LT(n_host_bytes, n_bytes);  // overflow

      // Copy host buffer
      new_impl->gidx_buffer =
          common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(n_host_bytes);
      dh::safe_cuda(cudaMemcpyAsync(new_impl->gidx_buffer.data(), old_impl->gidx_buffer.data(),
                                    n_host_bytes, cudaMemcpyDefault));

      // Copy compressed buffer
      dh::DeviceUVector<std::uint8_t> tmp;
      common::CompressEllpack(&ctx, old_impl->gidx_buffer.data() + n_host_bytes, n_compressed_bytes,
                              &tmp);
      // fixme: we should use tmp.size() here and fix the SizeBytes method and the
      // allocation in the Read method.
      auto c_page =
          common::MakeFixedVecWithPinnedMalloc<decltype(tmp)::value_type>(n_compressed_bytes);
      std::memset(c_page.data(), '\0', c_page.size_bytes());
      dh::safe_cuda(cudaMemcpyAsync(c_page.data(), tmp.data(), tmp.size(), cudaMemcpyDefault,
                                    ctx.CUDACtx()->Stream()));

      // Device buffer
      auto remaining = old_impl->gidx_buffer.size_bytes() - n_bytes;
      CHECK_GE(remaining, 0);
      auto d_page = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(remaining);
      if (remaining > 0) {
        dh::safe_cuda(cudaMemcpyAsync(d_page.data(), old_impl->gidx_buffer.data() + n_bytes,
                                      remaining, cudaMemcpyDefault));
      }

      CHECK_LE(new_impl->gidx_buffer.size(), old_impl->gidx_buffer.size());
      CHECK_EQ(new_impl->MemCostBytes() + d_page.size_bytes(), old_impl->MemCostBytes());
      LOG(INFO) << "Create cache page with size:"
                << common::HumanMemUnit(new_impl->MemCostBytes() + d_page.size_bytes());
      return std::make_tuple(std::move(new_impl), std::move(d_page), std::move(c_page));
    };
    if (no_concat) {
      // FIXME
      LOG(FATAL) << "not implemented";
      // Avoid a device->device->host copy.
      CHECK(new_page);
      auto new_impl = std::make_unique<EllpackPageImpl>();
      new_impl->CopyInfo(page.Impl());

      // Copy to host
      new_impl->gidx_buffer = common::MakeFixedVecWithPinnedMalloc<common::CompressedByteT>(
          page.Impl()->gidx_buffer.size());
      dh::safe_cuda(cudaMemcpyAsync(new_impl->gidx_buffer.data(), page.Impl()->gidx_buffer.data(),
                                    page.Impl()->gidx_buffer.size_bytes(), cudaMemcpyDefault));

      this->cache_->offsets.push_back(new_impl->n_rows * new_impl->info.row_stride);
      this->cache_->pages.push_back(std::move(new_impl));
      return new_page;
    }

    if (new_page) {
      // No need to copy if it's already in device.
      if (!this->cache_->pages.empty()) {
        // Need to wrap up the previous page.
        auto [commited, d_page, c_page] = commit_host_page(this->cache_->pages.back().get());
        // Replace the previous page (on device) with a new page on host.
        this->cache_->pages.back() = std::move(commited);
        this->cache_->d_pages.back() = std::move(d_page);
        this->cache_->c_pages.back() = std::move(c_page);
      }
      // Push a new page
      auto n_bytes = this->cache_->buffer_bytes.at(this->cache_->pages.size());
      auto n_samples = this->cache_->buffer_rows.at(this->cache_->pages.size());
      auto new_impl = std::make_unique<EllpackPageImpl>(&ctx, impl->CutsShared(), impl->IsDense(),
                                                        impl->info.row_stride, n_samples);
      new_impl->SetBaseRowId(impl->base_rowid);
      new_impl->SetNumSymbols(impl->NumSymbols());
      new_impl->gidx_buffer =
          common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(&ctx, n_bytes, 0);
      auto offset = new_impl->Copy(&ctx, impl, 0);

      this->cache_->offsets.push_back(offset);

      // Make sure we can always access the back of the vectors
      this->cache_->pages.push_back(std::move(new_impl));
      this->cache_->d_pages.emplace_back();
      this->cache_->c_pages.emplace_back();
    } else {
      // Concatenate into the device pages even though `d_pages` is used. We split the
      // page at the commit stage.
      CHECK(!this->cache_->pages.empty());
      CHECK_EQ(cache_idx, this->cache_->pages.size() - 1);
      auto& new_impl = this->cache_->pages.back();
      auto offset = new_impl->Copy(&ctx, impl, this->cache_->offsets.back());
      this->cache_->offsets.back() += offset;
    }

    // No need to copy if it's already in device.
    if (last_page) {
      auto [commited, d_page, c_page] = commit_host_page(this->cache_->pages.back().get());
      this->cache_->pages.back() = std::move(commited);
      this->cache_->d_pages.back() = std::move(d_page);
      this->cache_->c_pages.back() = std::move(c_page);
    }

    CHECK_EQ(this->cache_->pages.size(), this->cache_->d_pages.size());
    return new_page;
  }

  void Read(EllpackPage* out, bool prefetch_copy, curt::CUDAStreamView ds) const {

    CHECK_EQ(this->cache_->pages.size(), this->cache_->d_pages.size());
    auto const* page = this->cache_->At(this->ptr_);
    auto const& c_page = this->cache_->c_pages.at(this->ptr_);
    auto const& d_page = this->cache_->d_pages.at(this->ptr_);
    auto ctx = Context{}.MakeCUDA(dh::CurrentDevice());
    if (IsDevicePage(page)) {
      // Page is already in the device memory, no need to copy.
      prefetch_copy = false;
    }
    // FIXME: remove the device page cache and rely on the cache split instead.
    if (this->cache_->cache_host_ratio < 1.0) {
      prefetch_copy = true;
    }
    auto out_impl = out->Impl();
    if (prefetch_copy) {
      auto n = page->gidx_buffer.size() + d_page.size() + c_page.size();
      out_impl->gidx_buffer = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(n);

      // Copy host cache
      if (!page->gidx_buffer.empty()) {
        dh::safe_cuda(cudaMemcpyAsync(out_impl->gidx_buffer.data(), page->gidx_buffer.data(),
                                      page->gidx_buffer.size_bytes(), cudaMemcpyDefault,
                                      ctx.CUDACtx()->Stream()));
      }
      // Copy compressed host cache
      auto ptr = out_impl->gidx_buffer.data() + page->gidx_buffer.size();
      common::DecompressEllpack(ds, c_page.data(), ptr, c_page.size_bytes());

      // Copy device cache.
      if (!d_page.empty()) {
        auto beg = out_impl->gidx_buffer.data() + page->gidx_buffer.size() + c_page.size_bytes();
        dh::safe_cuda(cudaMemcpyAsync(beg, d_page.data(), d_page.size_bytes(), cudaMemcpyDefault,
                                      ctx.CUDACtx()->Stream()));
      }
    } else {
      // FIXME: Not implemented yet. Split is false as long as the host cache ratio is less than 1.
      CHECK(d_page.empty());
      auto res = page->gidx_buffer.Resource();
      out_impl->gidx_buffer = common::RefResourceView<common::CompressedByteT>{
          res->DataAs<common::CompressedByteT>(), page->gidx_buffer.size(), res};
    }

    out_impl->CopyInfo(page);
  }
};

/**
 * EllpackHostCacheStream
 */
EllpackHostCacheStream::EllpackHostCacheStream(std::shared_ptr<EllpackMemCache> cache)
    : p_impl_{std::make_unique<EllpackHostCacheStreamImpl>(std::move(cache))} {}

EllpackHostCacheStream::~EllpackHostCacheStream() = default;

std::shared_ptr<EllpackMemCache const> EllpackHostCacheStream::Share() const {
  return p_impl_->Share();
}

void EllpackHostCacheStream::Seek(bst_idx_t offset_bytes) { this->p_impl_->Seek(offset_bytes); }

void EllpackHostCacheStream::Read(EllpackPage* page, bool prefetch_copy,
                                  curt::CUDAStreamView ds) const {
  this->p_impl_->Read(page, prefetch_copy, ds);
}

[[nodiscard]] bool EllpackHostCacheStream::Write(EllpackPage const& page) {
  return this->p_impl_->Write(page);
}

/**
 * EllpackCacheStreamPolicy
 */
template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::WriterT>
EllpackCacheStreamPolicy<S, F>::CreateWriter(StringView, std::uint32_t iter) {
  if (!this->p_cache_) {
    this->p_cache_ = std::make_unique<EllpackMemCache>(this->CacheInfo());
  }
  auto fo = std::make_unique<EllpackHostCacheStream>(this->p_cache_);
  if (iter == 0) {
    CHECK(this->p_cache_->Empty());
  } else {
    fo->Seek(this->p_cache_->SizeBytes());
  }
  return fo;
}

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackCacheStreamPolicy<S, F>::ReaderT>
EllpackCacheStreamPolicy<S, F>::CreateReader(StringView, bst_idx_t offset, bst_idx_t) const {
  auto fi = std::make_unique<ReaderT>(this->p_cache_);
  fi->Seek(offset);
  return fi;
}

// Instantiation
template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::WriterT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateWriter(StringView name,
                                                                         std::uint32_t iter);

template std::unique_ptr<
    typename EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::ReaderT>
EllpackCacheStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateReader(StringView name,
                                                                         bst_idx_t offset,
                                                                         bst_idx_t length) const;

/**
 * EllpackMmapStreamPolicy
 */

template <typename S, template <typename> typename F>
[[nodiscard]] std::unique_ptr<typename EllpackMmapStreamPolicy<S, F>::ReaderT>
EllpackMmapStreamPolicy<S, F>::CreateReader(StringView name, bst_idx_t offset,
                                            bst_idx_t length) const {
  if (has_hmm_) {
    return std::make_unique<common::PrivateCudaMmapConstStream>(name, offset, length);
  } else {
    return std::make_unique<common::PrivateMmapConstStream>(name, offset, length);
  }
}

// Instantiation
template std::unique_ptr<
    typename EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>::ReaderT>
EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>::CreateReader(StringView name,
                                                                        bst_idx_t offset,
                                                                        bst_idx_t length) const;

void CalcCacheMapping(Context const* ctx, bool is_dense,
                      std::shared_ptr<common::HistogramCuts const> cuts,
                      std::int64_t min_cache_page_bytes, ExternalDataInfo const& ext_info,
                      EllpackCacheInfo* cinfo) {
  CHECK(cinfo->param.Initialized()) << "Need to initialize scalar fields first.";
  auto ell_info = CalcNumSymbols(ctx, ext_info.row_stride, is_dense, cuts);
  std::vector<std::size_t> cache_bytes;
  std::vector<std::size_t> cache_mapping(ext_info.n_batches, 0);
  std::vector<std::size_t> cache_rows;

  for (std::size_t i = 0; i < ext_info.n_batches; ++i) {
    auto n_samples = ext_info.base_rowids.at(i + 1) - ext_info.base_rowids[i];
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
  // if (cinfo->NumBatchesCc() == 1) {
  //   cinfo->cache_host_ratio = 0.0;  // FIXME: Add tests.
  //   LOG(INFO) << "Prefer device cache as there's only 1 page.";
  // }
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
    this->page_.reset(new EllpackPage{});
    auto* impl = this->page_->Impl();
    Context ctx = Context{}.MakeCUDA(this->Device().ordinal);
    if (this->GetCuts()->HasCategorical()) {
      CHECK(!this->feature_types_.empty());
    }
    *impl = EllpackPageImpl{&ctx, this->GetCuts(), *csr, is_dense_, row_stride_, feature_types_};
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
EllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();

/**
 * ExtEllpackPageSourceImpl
 */
template <typename F>
void ExtEllpackPageSourceImpl<F>::Fetch() {
  curt::SetDevice(this->Device().ordinal);
  if (!this->ReadCache()) {
    auto iter = this->source_->Iter();
    CHECK_EQ(this->Iter(), iter);
    cuda_impl::Dispatch(proxy_, [this](auto const& value) {
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
    LOG(INFO) << "Generated an Ellpack page with size: "
              << common::HumanMemUnit(this->page_->Impl()->MemCostBytes())
              << " from an batch with estimated size: "
              << cuda_impl::Dispatch<false>(proxy_, [](auto const& adapter) {
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
ExtEllpackPageSourceImpl<EllpackMmapStreamPolicy<EllpackPage, EllpackFormatPolicy>>::Fetch();
}  // namespace xgboost::data
