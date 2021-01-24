#include "iterative_device_dmatrix.h"
#include "adapter.h"
#include "rabit/rabit.h"
#include "proxy_dmatrix.h"
#include "../common/quantile.h"

namespace xgboost {
namespace data {

template <typename Fn>
decltype(auto) Dispatch(DMatrixProxy const* proxy, Fn fn) {
  if (proxy->Adapter().type() == typeid(std::shared_ptr<DenseAdapter>)) {
    auto value = dmlc::get<std::shared_ptr<DenseAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  } else {
    LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name();
    auto value = dmlc::get<std::shared_ptr<DenseAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  }
}

void IterativeDeviceDMatrix::InitializeHostData(DataIterHandle iter_handle, float missing, int nthread) {
  auto handle = static_cast<std::shared_ptr<DMatrix>*>(proxy_);
  CHECK(handle);
  DMatrixProxy* proxy = static_cast<DMatrixProxy*>(handle->get());
  CHECK(proxy);
  // The external iterator
  auto iter = DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{
    iter_handle, reset_, next_};

  auto num_rows = [&]() {
    return Dispatch(proxy, [](auto const &value) { return value.NumRows(); });
  };
  auto num_cols = [&]() {
    return Dispatch(proxy, [](auto const &value) { return value.NumCols(); });
  };

  bst_feature_t cols = 0;
  size_t accumulated_rows = 0;
  std::vector<common::HostSketchContainer> sketch_containers;
  while (iter.Next()) {
    if (cols == 0) {
      cols = num_cols();
      rabit::Allreduce<rabit::op::Max>(&cols, 1);
    } else {
      CHECK_EQ(cols, num_cols()) << "Inconsistent number of columns.";
    }

    auto const& info = proxy->Info();
    std::vector<bst_row_t> reduced(info.num_col_, 0);
    sketch_containers.emplace_back(reduced, batch_param_.max_bin,
                                   common::HostSketchContainer::UseGroup(info));
    Dispatch(proxy, [&](auto const &value) {
      sketch_containers.back().PushAdapterBatch(value, info);
    });
    auto batch_rows = num_rows();
    accumulated_rows += batch_rows;
  }

  iter.Reset();
  while (iter.Next()) {
  }
  iter.Reset();
  rabit::Allreduce<rabit::op::Max>(&info_.num_col_, 1);
}
}  // namespace data
}  // namespace xgboost
