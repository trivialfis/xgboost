/**
 * Copyright 2024, XGBoost Contributors
 */
#include <memory>   // for shared_ptr
#include <variant>  // for visit

#include "../common/cuda_rt_utils.h"  // for xgboost_NVTX_FN_RANGE
#include "batch_utils.h"              // for CheckParam, RegenGHist
#include "ellpack_page.cuh"           // for EllpackPage
#include "extmem_quantile_dmatrix.h"
#include "proxy_dmatrix.h"    // for DataIterProxy
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for BatchParam
#include "../common/cuda_rt_utils.h"

namespace xgboost::data {
void ExtMemQuantileDMatrix::InitFromCUDA(
    Context const *ctx,
    std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
    DMatrixHandle proxy_handle, BatchParam const &p, float missing, std::shared_ptr<DMatrix> ref) {
  xgboost_NVTX_FN_RANGE();

  // A handle passed to external iterator.
  auto proxy = MakeProxy(proxy_handle);
  CHECK(proxy);

  /**
   * Generate quantiles
   */
  auto cuts = std::make_shared<common::HistogramCuts>();
  ExternalDataInfo ext_info;
  cuda_impl::MakeSketches(ctx, iter.get(), proxy, ref, p, missing, cuts, this->Info(), &ext_info);
  ext_info.SetInfo(ctx, &this->info_);

  /**
   * Generate gradient index
   */
  auto id = MakeCache(this, ".ellpack.page", this->on_host_, cache_prefix_, &cache_info_);
  if (on_host_ && std::get_if<EllpackHostPtr>(&ellpack_page_source_) == nullptr) {
    ellpack_page_source_.emplace<EllpackHostPtr>(nullptr);
  }

  std::visit(
      [&](auto &&ptr) {
        using SourceT = typename std::remove_reference_t<decltype(ptr)>::element_type;
        // We can't hide the data load overhead for inference. Prefer device cache for
        // validation datasets.
        auto config = EllpackSourceConfig{.param = p,
                                          .prefer_device = (ref != nullptr),
                                          .missing = missing,
                                          .max_cache_page_ratio = this->max_cache_page_ratio_,
                                          .max_cache_ratio = this->max_device_cache_ratio_};
        ptr = std::make_shared<SourceT>(ctx, &this->Info(), ext_info, cache_info_.at(id), cuts,
                                        iter, proxy, config);
      },
      ellpack_page_source_);

  /**
   * Force initialize the cache and do some sanity checks along the way
   */
  bst_idx_t batch_cnt = 0, k = 0;
  bst_idx_t n_total_samples = 0;
  for (auto const &page : this->GetEllpackPageImpl()) {
    n_total_samples += page.Size();
    CHECK_EQ(page.Impl()->base_rowid, ext_info.base_rows[k]);
    CHECK_EQ(page.Impl()->info.row_stride, ext_info.row_stride);
    ++k, ++batch_cnt;
  }
  CHECK_EQ(batch_cnt, ext_info.n_batches);
  CHECK_EQ(n_total_samples, ext_info.accumulated_rows);

  this->n_batches_ = this->cache_info_.at(id)->Size();
}

[[nodiscard]] BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackPageImpl() {
  auto batch_set =
      std::visit([this](auto &&ptr) { return BatchSet{BatchIterator<EllpackPage>{ptr}}; },
                 this->ellpack_page_source_);
  return batch_set;
}

BatchSet<EllpackPage> ExtMemQuantileDMatrix::GetEllpackBatches(Context const *,
                                                               const BatchParam &param) {
  if (param.Initialized()) {
    detail::CheckParam(this->batch_, param);
    CHECK(!detail::RegenGHist(param, batch_)) << error::InconsistentMaxBin();
  }

  std::visit(
      [this, param](auto &&ptr) {
        CHECK(ptr)
            << "The `ExtMemQuantileDMatrix` is initialized using CPU data, cannot be used for GPU.";
        ptr->Reset(param);
      },
      this->ellpack_page_source_);

  return this->GetEllpackPageImpl();
}
}  // namespace xgboost::data
