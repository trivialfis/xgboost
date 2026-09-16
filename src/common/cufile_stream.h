/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <memory>   // for unique_ptr

#include "cuda_stream.h"          // for StreamRef
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::common {
// Load cuFile only when the disk cache is selected.
#if defined(XGBOOST_USE_CUDA)
void InitCuFile();
#else
inline void InitCuFile() { AssertGPUSupport(); }
#endif

/** @brief Aligned cache reader with stream-ordered cuFile reads into device memory. */
class CuFileReadStream {
  struct Impl;
  std::unique_ptr<Impl> impl_;

 public:
  CuFileReadStream(StringView path, std::size_t offset, std::size_t length);
  ~CuFileReadStream();

  [[nodiscard]] bool Read(void* ptr, std::size_t n_bytes);
  template <typename T>
  [[nodiscard]] bool Read(T* ptr) {
    return this->Read(ptr, sizeof(T));
  }
  [[nodiscard]] bool ReadAsync(void* ptr, std::size_t n_bytes, curt::StreamRef stream);
  // Wait for the device read and check its completion status before publishing the page.
  void Sync();
};
}  // namespace xgboost::common
