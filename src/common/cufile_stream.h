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

/** @brief Stream-ordered cuFile transfers between a cache file and device memory. */
class CuFileStream {
  struct Impl;
  std::unique_ptr<Impl> impl_;

 public:
  // A writer creates/truncates the file; a reader opens an existing file.
  explicit CuFileStream(StringView path, bool write = false);
  // Create independent request state sharing the registered file and IO stream.
  CuFileStream(CuFileStream const& other);
  ~CuFileStream();

  void ReadAsync(void* ptr, std::size_t n_bytes, std::size_t offset, curt::StreamRef stream);
  void WriteAsync(void const* ptr, std::size_t n_bytes, std::size_t offset, curt::StreamRef stream);
  // Check completion before publishing a page or reusing the request/buffer.
  void Sync();
};
}  // namespace xgboost::common
