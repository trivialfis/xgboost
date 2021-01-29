#include "iterative_device_dmatrix.h"
#include "simple_batch_iterator.h"

namespace xgboost {
namespace data {
namespace {
template <typename Fn>
decltype(auto) HostAdapterDispatch(DMatrixProxy const* proxy, Fn fn) {
  if (proxy->Adapter().type() == typeid(std::shared_ptr<CSRArrayAdapter>)) {
    auto value =
        dmlc::get<std::shared_ptr<CSRArrayAdapter>>(proxy->Adapter())->Value();
    return fn(value);
  } else if (proxy->Adapter().type() == typeid(std::shared_ptr<ArrayAdapter>)) {
    auto value = dmlc::get<std::shared_ptr<ArrayAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  } else {
    LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name();
    auto value = dmlc::get<std::shared_ptr<ArrayAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  }
}
}  // anonymous namespace

void IterativeDeviceDMatrix::InitializeExternalMemory(DataIterHandle iter_handle,
                                                      float missing,
                                                      int nthread) {
  auto proxy_handle = static_cast<std::shared_ptr<DMatrix>*>(proxy_);
  CHECK(proxy_handle) << "[xgboost::IterativeDMatrix] Invalid proxy handle.";
  DMatrixProxy* proxy = static_cast<DMatrixProxy*>(proxy_handle->get());
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_handle, reset_, next_};

  size_t n_batches = 0;
  size_t n_features = 0;
  size_t n_samples = 0;

  auto num_rows = [&]() {
    return HostAdapterDispatch(proxy, [](auto const &value) { return value.NumRows(); });
  };
  auto num_cols = [&]() {
    return HostAdapterDispatch(proxy, [](auto const &value) { return value.NumCols(); });
  };
  while (iter.Next()) {
    n_batches++;
    n_features = std::max(n_features, num_cols());
    n_samples += num_rows();
    this->info_.Extend(std::move(proxy->Info()), false);
  }
  iter.Reset();
  this->n_batches_ = n_batches;
  this->info_.num_row_ = n_samples;
  this->info_.num_col_ = n_features;
}

template <typename S>
class IterativeDMatrixIteratorImpl : public BatchIteratorImpl<S> {
 protected:
  DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter_;
  DMatrixProxy* proxy_;
  std::shared_ptr<S> page_;
  bool at_end_ {false};
  float missing_;
  int nthreads_;
  bst_feature_t n_features_;

 public:
  IterativeDMatrixIteratorImpl(
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter,
      DMatrixProxy *proxy, float missing, int nthreads, bst_feature_t n_features)
      : iter_{std::move(iter)}, proxy_{proxy}, missing_{missing},
        nthreads_{nthreads}, n_features_{n_features} {
    ++(*this);
  }

  S &operator*() override { return *page_; }
  const S &operator*() const override { return *page_; }
  std::shared_ptr<S> Page() const {
    return page_;
  }

  void operator++() override {
    at_end_ = !iter_.Next();
    if (at_end_) {
      iter_.Reset();
    } else {
      page_.reset(new S{});
      HostAdapterDispatch(proxy_, [&](auto const &value) {
        page_->Push(value, this->missing_, this->nthreads_);
      });
    }
  }
  bool AtEnd() const override { return at_end_; }
};

class IterativeDMatrixIteratorCSC : public IterativeDMatrixIteratorImpl<CSCPage> {
 public:
  using IterativeDMatrixIteratorImpl::IterativeDMatrixIteratorImpl;
  void operator++() override {
    at_end_ = !iter_.Next();
    if (at_end_) {
      iter_.Reset();
    } else {
      SparsePage page;
      HostAdapterDispatch(proxy_, [&](auto const &value) {
        page.Push(value, this->missing_, this->nthreads_);
        page_.reset(new CSCPage{page.GetTranspose(this->n_features_)});
      });
    }
  }
};

class IterativeDMatrixIteratorSortedCSC : public IterativeDMatrixIteratorImpl<SortedCSCPage> {
  using IterativeDMatrixIteratorImpl::IterativeDMatrixIteratorImpl;
  void operator++() override {
    at_end_ = !iter_.Next();
    if (at_end_) {
      iter_.Reset();
    } else {
      SparsePage page;
      HostAdapterDispatch(proxy_, [&](auto const &value) {
        page.Push(value, this->missing_, this->nthreads_);
        page = page.GetTranspose(this->n_features_);
        page.SortRows();
        page_.reset(new SortedCSCPage{std::move(page)});
      });
    }
  }
};

BatchSet<SparsePage> IterativeDeviceDMatrix::GetRowBatches() {
  auto proxy_handle = static_cast<std::shared_ptr<DMatrix>*>(proxy_);
  CHECK(proxy_handle) << "[xgboost::IterativeDMatrix] Invalid proxy handle.";
  DMatrixProxy* proxy = static_cast<DMatrixProxy*>(proxy_handle->get());
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  iter.Reset();
  auto ptr = new IterativeDMatrixIteratorImpl<SparsePage>(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_);
  sparse_page_ = ptr->Page();
  auto begin_iter = BatchIterator<SparsePage>(ptr);

  return BatchSet<SparsePage>(BatchIterator<SparsePage>(begin_iter));
}

BatchSet<CSCPage> IterativeDeviceDMatrix::GetColumnBatches() {
  auto proxy_handle = static_cast<std::shared_ptr<DMatrix>*>(proxy_);
  CHECK(proxy_handle) << "[xgboost::IterativeDMatrix] Invalid proxy handle.";
  DMatrixProxy* proxy = static_cast<DMatrixProxy*>(proxy_handle->get());
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  iter.Reset();
  auto begin_iter = BatchIterator<CSCPage>(new IterativeDMatrixIteratorCSC(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_));
  return BatchSet<CSCPage>(BatchIterator<CSCPage>(begin_iter));
}

BatchSet<SortedCSCPage> IterativeDeviceDMatrix::GetSortedColumnBatches() {
  auto proxy_handle = static_cast<std::shared_ptr<DMatrix>*>(proxy_);
  CHECK(proxy_handle) << "[xgboost::IterativeDMatrix] Invalid proxy handle.";
  DMatrixProxy* proxy = static_cast<DMatrixProxy*>(proxy_handle->get());
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  iter.Reset();
  CHECK_NE(this->Info().num_col_, 0);
  auto begin_iter = BatchIterator<SortedCSCPage>(new IterativeDMatrixIteratorSortedCSC(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_));
  return BatchSet<SortedCSCPage>(BatchIterator<SortedCSCPage>(begin_iter));
}

BatchSet<EllpackPage> IterativeDeviceDMatrix::GetEllpackBatches(const BatchParam& param) {
  if (!ellpack_page_) {
#if defined(XGBOOST_USE_CUDA)
    this->InitializeEllpack(iter_, missing_, nthreads_);
#endif  // defined(XGBOOST_USE_CUDA)
  }
  CHECK(ellpack_page_);
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_page_.get()));
  return BatchSet<EllpackPage>(begin_iter);
}
}  // namespace data
}  // namespace xgboost
