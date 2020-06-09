/*!
 * Copyright (c) by XGBoost Contributors 2020
 */
#ifndef XGBOOST_POOL_ALLOCATOR_H_
#define XGBOOST_POOL_ALLOCATOR_H_

#include <dmlc/thread_local.h>

#include <cinttypes>
#include <cstdlib>

namespace xgboost {
class MemoryPoolAllocator;
class WithPoolAllocator {
  using ThreadLocalAllocator = dmlc::ThreadLocalStore<MemoryPoolAllocator>;

  static int32_t& Context() {
    thread_local static int32_t context;
    return context;
  }

 public:
  WithPoolAllocator() {
    this->Context()++;
  }
  ~WithPoolAllocator();

  static void *Malloc(size_t size);
  static void Free(void* ptr);

  int32_t FrameDepth() const { return Context(); }

  bool InContext() const {
    return Context();
  }
};
}      // namespace xgboost
#endif  // XGBOOST_POOL_ALLOCATOR_H_
