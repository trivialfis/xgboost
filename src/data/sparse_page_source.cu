/**
 * Copyright 2021-2024, XGBoost contributors
 */
#include "../common/device_helpers.cuh"  // for CurrentDevice, DefaultStream
#include "proxy_dmatrix.cuh"             // for Dispatch, DMatrixProxy
#include "simple_dmatrix.cuh"            // for CopyToSparsePage
#include "sparse_page_source.h"
#include "xgboost/data.h"  // for SparsePage

namespace xgboost::data {
void DevicePush(DMatrixProxy *proxy, float missing, SparsePage *page) {
  auto device = proxy->Device();
  if (!device.IsCUDA()) {
    device = DeviceOrd::CUDA(dh::CurrentDevice());
  }
  CHECK(device.IsCUDA());

  cuda_impl::Dispatch(proxy,
                      [&](auto const &value) { CopyToSparsePage(value, device, missing, page); });
}

void InitNewThread::operator()() const {
  *GlobalConfigThreadLocalStore::Get() = config;
  dh::DefaultStream().Sync();  // Force initialize the global CUDA ctx
}
}  // namespace xgboost::data
