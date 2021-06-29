#include "iterative_device_dmatrix.h"
#include "simple_batch_iterator.h"
#include "sparse_page_source.h"

#include <future>
#include <thread>

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
  iter.Reset();

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
    n_features = std::max(n_features, num_cols());
    n_samples += num_rows();
    this->info_.Extend(std::move(proxy->Info()), false);
    n_batches++;
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

  size_t fetch_it_ {0};
  size_t it_{0};
  size_t n_batches_ {0};

  std::shared_ptr<Cache> cache_info_;
  using Queue = std::queue<std::future<std::shared_ptr<S>>>;
  std::unique_ptr<Queue> queue_ {new Queue};

  static size_t constexpr kPreFetch = 4;  // an heuristic for number of pre-fetched batches.

  bool ReadCache() {
    CHECK(!at_end_);
    CHECK_LT(it_, n_batches_);
    if (!cache_info_->written) {
      return false;
    }

    for (size_t i = 0; i < kPreFetch && fetch_it_ < n_batches_; ++i) {
      auto tloc_it = fetch_it_;
      auto future = std::async(std::launch::async, [this, tloc_it]() {
        std::unique_ptr<SparsePageFormat<S>> fmt{
            CreatePageFormat<S>(cache_info_->format)};
        std::unique_ptr<dmlc::SeekStream> fi{
            dmlc::SeekStream::CreateForRead(cache_info_->id.at(tloc_it).c_str())};
        auto page = std::make_shared<S>();
        fmt->Read(page.get(), fi.get());
        return page;
      });
      queue_->push(std::move(future));
      ++fetch_it_;
    }

    CHECK(!queue_->empty()) << ", it:" << fetch_it_;
    CHECK(queue_->front().valid());
    page_ = queue_->front().get();
    queue_->pop();
    return true;
  }

  void WriteCache() {
    if (!cache_info_->written) {
      std::unique_ptr<SparsePageFormat<S>> fmt{
          CreatePageFormat<S>(cache_info_->format)};
      std::unique_ptr<dmlc::Stream> fo_{
          dmlc::Stream::Create(cache_info_->id.at(fetch_it_).c_str(), "w")};
      fmt->Write(*page_, fo_.get());
    }
    fetch_it_++;
    CHECK_LE(fetch_it_, n_batches_);
  }

 public:
  IterativeDMatrixIteratorImpl(
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter,
      DMatrixProxy *proxy, float missing, int nthreads,
      bst_feature_t n_features, size_t n_batches, std::shared_ptr<Cache> cache)
      : iter_{std::move(iter)}, proxy_{proxy}, missing_{missing},
        nthreads_{nthreads}, n_features_{n_features}, n_batches_{n_batches},
        cache_info_{std::move(cache)} {}

  ~IterativeDMatrixIteratorImpl() override {
    while (!queue_->empty()) {
      auto& top = queue_->front();
      CHECK(top.valid());
      top.wait();
      top.get();
      queue_->pop();
    }
  }

  S &operator*() override {
    CHECK(page_);
    return *page_;
  }
  const S &operator*() const override {
    CHECK(page_);
    return *page_;
  }
  std::shared_ptr<S> Page() const {
    return page_;
  }

  void operator++() override = 0;
  bool AtEnd() const override {
    if (at_end_ && cache_info_->written) {
      CHECK_EQ(n_batches_, fetch_it_);
    }
    return at_end_;
  }

  void Reset() {
    iter_.Reset();
    ++(*this);
    fetch_it_ = 0;
    it_ = 0;
  }
};

class IterativeDMatrixIteratorCSR : public IterativeDMatrixIteratorImpl<SparsePage> {
 public:
  using IterativeDMatrixIteratorImpl::IterativeDMatrixIteratorImpl;
  void operator++() override {
    at_end_ = !iter_.Next() || it_ == n_batches_;
    if (at_end_) {
      cache_info_->written = true;
    } else {
      CHECK_LT(it_, n_batches_);
      page_.reset(new SparsePage{});
      if (!this->ReadCache()) {
        HostAdapterDispatch(proxy_, [&](auto const &value) {
          page_->Push(value, this->missing_, this->nthreads_);
        });
        this->WriteCache();
      }
      it_ ++;
    }
  }
};

class IterativeDMatrixIteratorCSC : public IterativeDMatrixIteratorImpl<CSCPage> {
  std::shared_ptr<IterativeDMatrixIteratorCSR> csr_;

 public:
  IterativeDMatrixIteratorCSC(
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter,
      DMatrixProxy *proxy, float missing, int nthreads,
      bst_feature_t n_features, size_t n_batches, std::shared_ptr<Cache> cache,
      std::shared_ptr<IterativeDMatrixIteratorCSR> csr)
      : IterativeDMatrixIteratorImpl(iter, proxy, missing, nthreads, n_features,
                                     n_batches, cache),
        csr_{std::move(csr)} {}

  void operator++() override {
    if (csr_) {
      ++(*csr_);
      at_end_ = csr_->AtEnd();
    } else {
      at_end_ = !iter_.Next();
    }
    at_end_ = at_end_ || it_ == n_batches_;

    if (at_end_) {
      cache_info_->written = true;
    } else {
      if (!this->ReadCache()) {
        if (csr_) {
          auto page = csr_->Page();
          CSCPage sorted{page->GetTranspose(this->n_features_)};
          page_.reset(new CSCPage(std::move(sorted)));
        } else {
          HostAdapterDispatch(proxy_, [&](auto const &value) {
            SparsePage page;
            page.Push(value, this->missing_, this->nthreads_);
            page_.reset(new CSCPage{page.GetTranspose(this->n_features_)});
          });
        }
        this->WriteCache();
      }
      it_++;
    }
  }
};

class IterativeDMatrixIteratorSortedCSC
    : public IterativeDMatrixIteratorImpl<SortedCSCPage> {
  std::shared_ptr<IterativeDMatrixIteratorCSR> csr_;

 public:
  IterativeDMatrixIteratorSortedCSC(
      DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> iter,
      DMatrixProxy *proxy, float missing, int nthreads,
      bst_feature_t n_features, size_t n_batches, std::shared_ptr<Cache> cache,
      std::shared_ptr<IterativeDMatrixIteratorCSR> csr)
      : IterativeDMatrixIteratorImpl(iter, proxy, missing, nthreads, n_features,
                                     n_batches, cache),
        csr_{std::move(csr)} {}

  void operator++() override {
    if (csr_) {
      at_end_ = csr_->AtEnd();
    } else {
      at_end_ = !iter_.Next();
    }
    at_end_ = at_end_ || it_ == n_batches_;

    if (at_end_) {
      cache_info_->written = true;
    } else {
      if (!this->ReadCache()) {
        if (csr_) {
          ++(*csr_);
          auto page = csr_->Page();
          SortedCSCPage sorted{page->GetTranspose(this->n_features_)};
          sorted.SortRows();
          page_.reset(new SortedCSCPage(std::move(sorted)));
        } else {
          HostAdapterDispatch(proxy_, [&](auto const &value) {
            SparsePage page;
            page.Push(value, this->missing_, this->nthreads_);
            page = page.GetTranspose(this->n_features_);
            page.SortRows();
            page_.reset(new SortedCSCPage{std::move(page)});
          });
        }
        this->WriteCache();
      }
      it_++;
    }
  }
};

namespace {
DMatrixProxy *MakeProxy(DMatrixHandle proxy) {
  auto proxy_handle = static_cast<std::shared_ptr<DMatrix> *>(proxy);
  CHECK(proxy_handle) << "[xgboost::IterativeDMatrix] Invalid proxy handle.";
  DMatrixProxy *typed = static_cast<DMatrixProxy *>(proxy_handle->get());
  return typed;
}

[[nodiscard]] std::string MakeId(IterativeDeviceDMatrix *ptr,
                                 std::string format) {
  std::stringstream ss;
  ss << ptr;
  auto id = ss.str() + format;
  return id;
}

[[nodiscard]] std::string MakeShardName(IterativeDeviceDMatrix *ptr, size_t i,
                                        std::string format) {
  std::stringstream ss;
  ss << ptr << "-" << i;
  auto id = ss.str() + format;
  return id;
}

[[nodiscard]] std::string
MakeCache(IterativeDeviceDMatrix *ptr, std::string format, size_t n_batches,
          std::map<std::string, std::shared_ptr<Cache>> *out) {
  auto& cache_info = *out;
  auto id = MakeId(ptr, format);
  auto it = cache_info.find(id);
  CHECK_GT(n_batches, 0);
  if (it == cache_info.cend()) {
    CHECK(!CheckCacheFileExists(id));
    std::vector<std::string> names;
    for (size_t i = 0; i < n_batches; ++i) {
      names.emplace_back(MakeShardName(ptr, i, format));
    }
    cache_info[id].reset(new Cache{false, "raw", std::move(names)});
  }
  return id;
}
} // namespace

void IterativeDeviceDMatrix::InitSparseSource() {
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  iter.Reset();
  auto id = MakeCache(this, ".row.page", this->n_batches_, &cache_info_);
  sparse_source_.reset(new IterativeDMatrixIteratorCSR(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_,
      this->n_batches_, cache_info_.at(id)));
  sparse_source_->Reset();
}

BatchSet<SparsePage> IterativeDeviceDMatrix::GetRowBatches() {
  this->InitSparseSource();
  auto begin_iter = BatchIterator<SparsePage>(sparse_source_);
  return BatchSet<SparsePage>(BatchIterator<SparsePage>(begin_iter));
}

BatchSet<CSCPage> IterativeDeviceDMatrix::GetColumnBatches() {
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  iter.Reset();
  auto id = MakeCache(this, ".col.page", this->n_batches_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  if (!lazy_) { this->InitSparseSource(); }

  auto ptr = std::make_shared<IterativeDMatrixIteratorCSC>(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_,
      this->n_batches_, cache_info_.at(id), sparse_source_);
  ptr->Reset();
  auto begin_iter = BatchIterator<CSCPage>(ptr);
  return BatchSet<CSCPage>(BatchIterator<CSCPage>(begin_iter));
}

BatchSet<SortedCSCPage> IterativeDeviceDMatrix::GetSortedColumnBatches() {
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  iter.Reset();
  auto id = MakeCache(this, ".sorted.col.page", this->n_batches_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  if (!lazy_) { this->InitSparseSource(); }
  auto ptr = std::make_shared<IterativeDMatrixIteratorSortedCSC>(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_,
      this->n_batches_, cache_info_.at(id), sparse_source_);
  ptr->Reset();
  auto begin_iter = BatchIterator<SortedCSCPage>(ptr);
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
