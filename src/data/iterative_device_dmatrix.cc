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

  size_t it_ {0};
  size_t n_batches_ {0};

  std::shared_ptr<Cache> cache_info_;
  using Queue = std::queue<std::future<std::shared_ptr<S>>>;
  std::unique_ptr<Queue> queue_ {new Queue};

  static size_t constexpr kPreFetch = 4;  // an heuristic for number of pre-fetched batches.

  bool ReadCache() {
    CHECK(!at_end_);
    if (!cache_info_->written) {
      return false;
    }

    for (size_t i = 0; i < kPreFetch && it_ < n_batches_; ++i) {
      auto tloc_it = it_;
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
      ++it_;
    }

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
          dmlc::Stream::Create(cache_info_->id.at(it_).c_str(), "w")};
      fmt->Write(*page_, fo_.get());
    }
    it_++;
    CHECK_LE(it_, n_batches_);
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
      CHECK_EQ(n_batches_, it_);
    }
    return at_end_;
  }

  void Reset() {
    iter_.Reset();
    ++(*this);
  }
};

class IterativeDMatrixIteratorCSR : public IterativeDMatrixIteratorImpl<SparsePage> {
 public:
  using IterativeDMatrixIteratorImpl::IterativeDMatrixIteratorImpl;
  void operator++() override {
    at_end_ = !iter_.Next();
    if (at_end_) {
      cache_info_->written = true;
    } else {
      page_.reset(new SparsePage{});
      if (!this->ReadCache()) {
        HostAdapterDispatch(proxy_, [&](auto const &value) {
          page_->Push(value, this->missing_, this->nthreads_);
        });
        this->WriteCache();
      }
    }
  }
};

class IterativeDMatrixIteratorCSC : public IterativeDMatrixIteratorImpl<CSCPage> {
 public:
  using IterativeDMatrixIteratorImpl::IterativeDMatrixIteratorImpl;
  void operator++() override {
    at_end_ = !iter_.Next();
    if (at_end_) {
      cache_info_->written = true;
    } else {
      if (!this->ReadCache()) {
        HostAdapterDispatch(proxy_, [&](auto const &value) {
          SparsePage page;
          page.Push(value, this->missing_, this->nthreads_);
          page_.reset(new CSCPage{page.GetTranspose(this->n_features_)});
        });
        this->WriteCache();
      }
    }
  }
};

class IterativeDMatrixIteratorSortedCSC : public IterativeDMatrixIteratorImpl<SortedCSCPage> {
  using IterativeDMatrixIteratorImpl::IterativeDMatrixIteratorImpl;
  void operator++() override {
    at_end_ = !iter_.Next();
    if (at_end_) {
      cache_info_->written = true;
    } else {
      if (!this->ReadCache()) {
        HostAdapterDispatch(proxy_, [&](auto const &value) {
          SparsePage page;
          page.Push(value, this->missing_, this->nthreads_);
          page = page.GetTranspose(this->n_features_);
          page.SortRows();
          page_.reset(new SortedCSCPage{std::move(page)});
        });
        this->WriteCache();
      }
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

std::string MakeId(IterativeDeviceDMatrix* ptr, std::string format) {
  std::stringstream ss;
  ss << ptr;
  auto id = ss.str() + format;
  return id;
}

std::string MakeShardName(IterativeDeviceDMatrix* ptr, size_t i, std::string format) {
  std::stringstream ss;
  ss << ptr << "-" << i;
  auto id = ss.str() + format;
  return id;
}

std::string MakeCache(IterativeDeviceDMatrix *ptr, std::string format,
                      size_t n_batches,
                      std::map<std::string, std::shared_ptr<Cache>> *out) {
  auto& cache_info = *out;
  auto id = MakeId(ptr, format);
  auto it = cache_info.find(id);
  CHECK_GT(n_batches, 0);
  if (it == cache_info.cend()) {
    std::vector<std::string> names;
    for (size_t i = 0; i < n_batches; ++i) {
      names.emplace_back(MakeShardName(ptr, i, format));
    }
    cache_info[id].reset(new Cache{false, "raw", std::move(names)});
  }
  return id;
}
} // namespace

BatchSet<SparsePage> IterativeDeviceDMatrix::GetRowBatches() {
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  auto id = MakeCache(this, ".row.page", this->n_batches_, &cache_info_);
  auto ptr = new IterativeDMatrixIteratorCSR(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_,
      this->n_batches_, cache_info_.at(id));
  sparse_page_ = ptr->Page();
  ptr->Reset();
  auto begin_iter = BatchIterator<SparsePage>(ptr);
  return BatchSet<SparsePage>(BatchIterator<SparsePage>(begin_iter));
}

BatchSet<CSCPage> IterativeDeviceDMatrix::GetColumnBatches() {
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  auto id = MakeCache(this, ".col.page", this->n_batches_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  auto ptr = new IterativeDMatrixIteratorCSC(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_,
      this->n_batches_, cache_info_.at(id));
  auto begin_iter = BatchIterator<CSCPage>(ptr);
  return BatchSet<CSCPage>(BatchIterator<CSCPage>(begin_iter));
}

BatchSet<SortedCSCPage> IterativeDeviceDMatrix::GetSortedColumnBatches() {
  DMatrixProxy *proxy = MakeProxy(proxy_);
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
      iter_, reset_, next_};
  auto id = MakeCache(this, ".sorted.col.page", this->n_batches_, &cache_info_);
  CHECK_NE(this->Info().num_col_, 0);
  auto ptr = new IterativeDMatrixIteratorSortedCSC(
      iter, proxy, this->missing_, this->nthreads_, this->Info().num_col_,
      this->n_batches_, cache_info_.at(id));
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
