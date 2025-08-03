/**
 * Copyright 2025, XGBoost Contributors
 */
#include <thrust/copy.h>  // for copy

#include <memory>  // for make_unique
#include <vector>  // for vector

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for ToSpan
#include "../common/device_vector.cuh"   // for device_vector
#include "../common/type.h"              // for GetValueT, EraseType
#include "../data/array_interface.h"     // for IsCudaPtr
#include "../encoder/ordinal.cuh"        // for SortNames
#include "../encoder/ordinal.h"          // for DictionaryView
#include "../encoder/types.h"            // for Overloaded
#include "cat_container.cuh"             // for CatStrArray
#include "cat_container.h"               // for CatContainer
#include "xgboost/span.h"                // for Span

namespace xgboost {
namespace cuda_impl {
namespace {
template <typename Fn, typename Col>
decltype(auto) Visit(Fn&& dispatch, Col const& col) {
  using ColT = common::GetValueT<decltype(col)>;
  if constexpr (std::is_same_v<ColT, enc::HostCatIndexView>) {
    return std::visit(dispatch, col);
  } else {
    static_assert(std::is_same_v<ColT, enc::DeviceCatIndexView>);
    return cuda::std::visit(dispatch, col);
  }
}
}  // namespace

struct CatContainerImpl {
  dh::device_vector<enc::DeviceCatIndexView> columns_v;

  TableCatStorage storage;

  template <
      typename VariantT,
      typename Columns = decltype(std::declval<enc::detail::ColumnsViewImpl<VariantT>>().columns)>
  void CopyFromImpl(Context const* ctx, Columns const& that) {
    CHECK(!ArrayInterfaceHandler::IsCudaPtr(that.data()));
    auto d_data = common::EraseType(dh::ToSpan(this->storage.data));
    auto d_offsets = common::EraseType(dh::ToSpan(this->storage.offsets));

    // Gather all the pointers for batch copy.
    std::vector<void const*> src_ptrs;
    std::vector<std::size_t> sizes;
    std::vector<void*> dst_ptrs;

    for (std::size_t f_idx = 0, n = that.size(); f_idx < n; ++f_idx) {
      auto const& col_v = that[f_idx];
      std::size_t dst_off = 0;
      Visit(enc::Overloaded{
                [&](enc::CatStrArrayView const& str) {
                  auto p_off = str.offsets.data();
                  auto p_data = str.values.data();

                  src_ptrs.push_back(p_off);
                  src_ptrs.push_back(p_data);

                  sizes.push_back(str.values.size_bytes());
                  sizes.push_back(str.offsets.size_bytes());

                  dst_ptrs.push_back(d_data.subspan(dst_off, str.values.size_bytes()).data());
                  dst_off += str.values.size_bytes();
                  dst_ptrs.push_back(d_offsets.subspan(dst_off, str.offsets.size_bytes()).data());
                  dst_off += str.offsets.size_bytes();
                },
                [&](auto&& values) {
                  src_ptrs.push_back(values.data());
                  sizes.push_back(values.size_bytes());

                  dst_ptrs.push_back(d_data.subspan(dst_off, values.size_bytes()).data());
                  dst_off += values.size_bytes();
                }},
            col_v);
    }

    // Copy into the container
    std::size_t fail_idx = 0;
    if constexpr (std::is_same_v<enc::HostColumnsView, decltype(that)>) {
      auto status = dh::MemcpyBatchAsync<cudaMemcpyHostToDevice>(
          dst_ptrs.data(), src_ptrs.data(), sizes.data(), sizes.size(), &fail_idx,
          ctx->CUDACtx()->Stream());
      dh::safe_cuda(status);
    } else {
      auto status = dh::MemcpyBatchAsync<cudaMemcpyDeviceToDevice>(
          dst_ptrs.data(), src_ptrs.data(), sizes.data(), sizes.size(), &fail_idx,
          ctx->CUDACtx()->Stream());
      dh::safe_cuda(status);
    }

    // Construct the views
    std::vector<decltype(columns_v)::value_type> h_columns_v(this->columns_v.size());
    for (std::size_t f_idx = 0, n = that.size(); f_idx < n; ++f_idx) {
      std::size_t ptr_idx = 0;
      Visit(enc::Overloaded{
                [&](enc::CatStrArrayView const& str) {
                  auto n = sizes[ptr_idx];
                  CHECK_EQ(n, str.values.size_bytes());
                  auto ptr = dst_ptrs[ptr_idx];
                  ptr_idx += 1;
                  static_assert(sizeof(enc::CatCharT) == 1);

                  using OffT = decltype(std::declval<enc::CatStrArrayView>().offsets)::value_type;

                  auto ptr_off = dst_ptrs[ptr_idx];
                  n = sizes[ptr_idx];
                  CHECK_EQ(n, str.offsets.size_bytes());
                  auto n_off = n / sizeof(OffT);
                  CHECK_EQ(n_off, str.offsets.size());
                  ptr_idx += 1;

                  h_columns_v[f_idx].emplace<enc::CatStrArrayView>();
                  auto& col_v = cuda::std::get<enc::CatStrArrayView>(h_columns_v[f_idx]);
                  col_v = {common::Span{static_cast<OffT const*>(ptr_off), n_off},
                           common::Span{static_cast<enc::CatCharT const*>(ptr), n}};
                },
                [&](auto&& values) {
                  using T = std::remove_cv_t<typename std::decay_t<decltype(values)>::value_type>;
                  using V = common::Span<std::add_const_t<T>>;
                  h_columns_v[f_idx].emplace<V>();

                  auto ptr = dst_ptrs[ptr_idx];
                  CHECK_EQ(values.size_bytes(), sizes[ptr_idx]);

                  auto& col_v = cuda::std::get<V>(h_columns_v[f_idx]);
                  col_v = common::Span{static_cast<T const*>(ptr), values.size_bytes()};

                  ptr_idx += 1;
                }},
            that[f_idx]);
    }

    dh::safe_cuda(cudaMemcpyAsync(thrust::raw_pointer_cast(columns_v.data()), h_columns_v.data(),
                                  dh::ToSpan(columns_v).size_bytes(), cudaMemcpyDefault,
                                  ctx->CUDACtx()->Stream()));
  }

  void CopyFrom(Context const* ctx, CatContainerImpl const* that) {
    this->storage.data.resize(that->storage.data.size());
    this->storage.offsets.resize(that->storage.offsets.size());
    this->columns_v.resize(that->columns_v.size());

    std::vector<decltype(columns_v)::value_type> h_columns_v(that->columns_v.size());
    dh::safe_cuda(cudaMemcpyAsync(
        h_columns_v.data(), thrust::raw_pointer_cast(that->columns_v.data()),
        dh::ToSpan(columns_v).size_bytes(), cudaMemcpyDefault, ctx->CUDACtx()->Stream()));

    this->CopyFromImpl<enc::DeviceCatIndexView>(ctx, common::Span{h_columns_v});
  }

  template <typename VariantT>  // fixme: doesn't handle host
  void CopyFrom(Context const* ctx, enc::detail::ColumnsViewImpl<VariantT> that) {
    this->columns_v.resize(that.columns.size());

    std::size_t n_bytes = 0;
    for (auto const& col_v : that.columns) {
      n_bytes += Visit([&](auto&& values) { return values.size_bytes(); }, col_v);
    }

    this->storage.data.resize(n_bytes);
    this->storage.offsets.resize(that.columns.size() + 1);

    this->CopyFromImpl<VariantT>(ctx, that.columns);
  }

  void CopyTo(cpu_impl::CatContainerImpl* that) const {
    that->columns_v.clear();
    that->columns.clear();

    std::vector<decltype(columns_v)::value_type> h_columns_v(this->columns_v.size());
    dh::safe_cuda(cudaMemcpyAsync(h_columns_v.data(), h_columns_v.data(),
                                  common::Span{h_columns_v}.size_bytes(), cudaMemcpyDefault));

    // Gather all the pointers for batch copy.
    std::vector<void const*> src_ptrs;
    std::vector<std::size_t> sizes;
    std::vector<void*> dst_ptrs;

    for (auto const& col : h_columns_v) {
      that->columns.emplace_back();
      auto& out_col = that->columns.back();
      cuda::std::visit(
          enc::Overloaded{[&](enc::CatStrArrayView const& str) {
                            out_col.emplace<cpu_impl::CatStrArray>();
                            auto& out_str = std::get<cpu_impl::CatStrArray>(out_col);
                            // Offsets
                            out_str.offsets.resize(str.offsets.size());
                            if (!out_str.offsets.empty()) {
                              src_ptrs.push_back(str.offsets.data());
                              dst_ptrs.push_back(out_str.offsets.data());
                              sizes.push_back(common::Span{out_str.offsets}.size_bytes());
                            }
                            // Values
                            out_str.values.resize(str.values.size());
                            if (!out_str.values.empty()) {
                              src_ptrs.push_back(str.values.data());
                              dst_ptrs.push_back(out_str.values.data());
                              sizes.push_back(common::Span{out_str.values}.size_bytes());
                            }
                          },
                          [&](auto&& values) {
                            using T0 = decltype(values);
                            using T1 = std::add_const_t<typename std::decay_t<T0>::value_type>;
                            using Vec =
                                typename cpu_impl::ViewToStorageImpl<common::Span<T1>>::Type;
                            out_col.emplace<Vec>();
                            auto& out_vec = std::get<Vec>(out_col);
                            out_vec.resize(values.size());
                            if (!out_vec.empty()) {
                              src_ptrs.push_back(values.data());
                              dst_ptrs.push_back(out_vec.data());
                              sizes.push_back(common::Span{out_vec}.size_bytes());
                            }
                          }},
          col);
    }
    std::size_t fail_idx = 0;
    auto status =
        dh::MemcpyBatchAsync<cudaMemcpyDeviceToHost>(dst_ptrs.data(), src_ptrs.data(), sizes.data(),
                                                     sizes.size(), &fail_idx, dh::DefaultStream());
    dh::safe_cuda(status);
    that->Finalize();
  }
};
}  // namespace cuda_impl

CatContainer::CatContainer()  // NOLINT
    : cpu_impl_{std::make_unique<cpu_impl::CatContainerImpl>()},
      cu_impl_{std::make_unique<cuda_impl::CatContainerImpl>()} {}

CatContainer::CatContainer(Context const* ctx, enc::DeviceColumnsView const& df, bool is_ref)
    : CatContainer{} {
  this->is_ref_ = is_ref;
  this->n_total_cats_ = df.n_total_cats;

  this->feature_segments_.SetDevice(ctx->Device());
  this->feature_segments_.Resize(df.feature_segments.size());
  auto d_segs = this->feature_segments_.DeviceSpan();
  thrust::copy_n(ctx->CUDACtx()->CTP(), dh::tcbegin(df.feature_segments),
                 df.feature_segments.size(), dh::tbegin(d_segs));

  // FIXME(jiamingy): We can use a single kernel for copying data once cuDF can return
  // device data. Remove this along with the one in the device cuDF adapter.
  this->cu_impl_->CopyFrom(ctx, df);

  this->sorted_idx_.SetDevice(ctx->Device());
  this->sorted_idx_.Resize(0);
  if (this->n_total_cats_ > 0) {
    CHECK(this->DeviceCanRead());
    CHECK(!this->HostCanRead());
    CHECK(!this->cu_impl_->columns_v.empty());
  }
}

CatContainer::~CatContainer() = default;

void CatContainer::Copy(Context const* ctx, CatContainer const& that) {
  if (ctx->IsCPU()) {
    // Pull data to host
    [[maybe_unused]] auto h_view = that.HostView();
    this->CopyCommon(ctx, that);
    this->cpu_impl_->Copy(that.cpu_impl_.get());
    CHECK(!this->DeviceCanRead());
  } else {
    // Pull data to device
    [[maybe_unused]] auto d_view = that.DeviceView(ctx);
    this->CopyCommon(ctx, that);
    this->cu_impl_->CopyFrom(ctx, that.cu_impl_.get());
    CHECK(this->Empty() || !this->HostCanRead());
  }
  if (ctx->IsCPU()) {
    CHECK_EQ(this->cpu_impl_->columns_v.size(), that.cpu_impl_->columns_v.size());
    CHECK_EQ(this->cpu_impl_->columns.size(), that.cpu_impl_->columns.size());
    CHECK(this->HostCanRead());
  } else {
    CHECK_EQ(this->cu_impl_->columns_v.size(), that.cu_impl_->columns_v.size());
    // CHECK_EQ(this->cu_impl_->columns.size(), that.cu_impl_->columns.size());
    CHECK(this->DeviceCanRead());
  }
  CHECK_EQ(this->Empty(), that.Empty());
  CHECK_EQ(this->NumCatsTotal(), that.NumCatsTotal());
}

[[nodiscard]] bool CatContainer::Empty() const {
  return this->HostCanRead() ? this->cpu_impl_->columns.empty() : this->cu_impl_->columns_v.empty();
}

[[nodiscard]] std::size_t CatContainer::NumFeatures() const {
  if (this->HostCanRead()) {
    return this->cpu_impl_->columns.size();
  }
  return this->cu_impl_->columns_v.size();
}

void CatContainer::Sort(Context const* ctx) {
  if (!this->HasCategorical()) {
    return;
  }

  if (ctx->IsCPU()) {
    auto view = this->HostView();
    CHECK(!view.Empty()) << view.n_total_cats;
    this->sorted_idx_.HostVector().resize(view.n_total_cats);
    enc::SortNames(cpu_impl::EncPolicy, view, this->sorted_idx_.HostSpan());
  } else {
    auto view = this->DeviceView(ctx);
    CHECK(!view.Empty()) << view.n_total_cats;
    this->sorted_idx_.SetDevice(ctx->Device());
    this->sorted_idx_.Resize(view.n_total_cats);
    enc::SortNames(cuda_impl::EncPolicy, view, this->sorted_idx_.DeviceSpan());
  }
}

[[nodiscard]] enc::HostColumnsView CatContainer::HostView() const {
  std::lock_guard guard{device_mu_};
  if (!this->HostCanRead()) {
    this->feature_segments_.ConstHostSpan();
    // Lazy copy to host
    this->cu_impl_->CopyTo(this->cpu_impl_.get());
  }
  CHECK(this->HostCanRead());
  return this->HostViewImpl();
}

[[nodiscard]] enc::DeviceColumnsView CatContainer::DeviceView(Context const* ctx) const {
  CHECK(ctx->IsCUDA());
  std::lock_guard guard{device_mu_};
  if (!this->DeviceCanRead()) {
    this->feature_segments_.SetDevice(ctx->Device());
    this->feature_segments_.ConstDeviceSpan();
    // Lazy copy to device
    auto h_view = this->HostViewImpl();
    this->cu_impl_->CopyFrom(ctx, h_view);
    CHECK_EQ(this->cu_impl_->columns_v.size(), this->cpu_impl_->columns_v.size());
    CHECK_EQ(this->cu_impl_->columns_v.size(), this->cpu_impl_->columns.size());
  }
  CHECK(this->DeviceCanRead());
  if (this->n_total_cats_ != 0) {
    CHECK(!this->cu_impl_->columns_v.empty());
    CHECK_EQ(this->feature_segments_.Size(), this->cu_impl_->columns_v.size() + 1);
  }
  return {dh::ToSpan(this->cu_impl_->columns_v), this->feature_segments_.ConstDeviceSpan(),
          this->n_total_cats_};
}
}  // namespace xgboost
