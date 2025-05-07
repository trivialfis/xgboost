/**
 * Copyright 2024-2025, XGBoost contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t

#include "common.h"

#if defined(XGBOOST_USE_CUDA)
#include <cuda_runtime_api.h>
#endif  // defined(XGBOOST_USE_CUDA)

namespace xgboost::curt {
std::int32_t AllVisibleGPUs();

/**
 * @param raise Raise error if XGBoost is not compiled with CUDA, or GPU is not available.
 */
std::int32_t CurrentDevice(bool raise = true);

// Whether the device supports coherently accessing pageable memory without calling
// `cudaHostRegister` on it
bool SupportsPageableMem();

// Address Translation Service (ATS)
bool SupportsAts();

void CheckComputeCapability();

void SetDevice(std::int32_t device);

/**
 * @brief Total device memory size.
 */
[[nodiscard]] std::size_t TotalMemory();

// Returns the CUDA Runtime version.
void RtVersion(std::int32_t *major, std::int32_t *minor);

// Returns the latest version of CUDA supported by the driver.
void DrVersion(std::int32_t *major, std::int32_t *minor);

class CUDAStreamView;

class CUDAEvent {
  std::unique_ptr<cudaEvent_t, void (*)(cudaEvent_t *)> event_;

 public:
  CUDAEvent()
      : event_{[] {
                 auto e = new cudaEvent_t;
                 dh::safe_cuda(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
                 return e;
               }(),
               [](cudaEvent_t *e) {
                 if (e) {
                   dh::safe_cuda(cudaEventDestroy(*e));
                   delete e;
                 }
               }} {}

  inline void Record(CUDAStreamView stream);  // NOLINT
  // Define swap-based ctor to make sure an event is always valid.
  CUDAEvent(CUDAEvent &&e) : CUDAEvent() { std::swap(this->event_, e.event_); }
  CUDAEvent &operator=(CUDAEvent &&e) {
    std::swap(this->event_, e.event_);
    return *this;
  }

  operator cudaEvent_t() const { return *event_; }                // NOLINT
  cudaEvent_t const *data() const { return this->event_.get(); }  // NOLINT
};

class CUDAStreamView {
  cudaStream_t stream_{nullptr};

 public:
  explicit CUDAStreamView(cudaStream_t s) : stream_{s} {}
  void Wait(CUDAEvent const &e) {
#if defined(__CUDACC_VER_MAJOR__)
#if __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 0
    // CUDA == 11.0
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, 0));
#else
    // CUDA > 11.0
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, cudaEventWaitDefault));
#endif  // __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ == 0:
#else   // clang
    dh::safe_cuda(cudaStreamWaitEvent(stream_, cudaEvent_t{e}, cudaEventWaitDefault));
#endif  //  defined(__CUDACC_VER_MAJOR__)
  }
  operator cudaStream_t() const {  // NOLINT
    return stream_;
  }
  cudaError_t Sync(bool error = true) {
    if (error) {
      dh::safe_cuda(cudaStreamSynchronize(stream_));
      return cudaSuccess;
    }
    return cudaStreamSynchronize(stream_);
  }
};

class CUDAStream {
  cudaStream_t stream_;

 public:
  CUDAStream() { dh::safe_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~CUDAStream() { dh::safe_cuda(cudaStreamDestroy(stream_)); }

  [[nodiscard]] CUDAStreamView View() const { return CUDAStreamView{stream_}; }
  [[nodiscard]] cudaStream_t Handle() const { return stream_; }

  void Sync() { this->View().Sync(); }
  void Wait(CUDAEvent const &e) { this->View().Wait(e); }
};
}  // namespace xgboost::curt
