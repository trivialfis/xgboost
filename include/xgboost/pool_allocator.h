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
/* \brief A global context class for using pool allocator.
 *
 * Example usage:
 *
 * \code
 *   {
 *      WithPoolAllocator allocator_ctx;
 *      WithPoolAllocator::Malloc(100);  // This will use the global pool allocator;
 *   }
 *
 *  // On any other time, this is equivalent to calling `std::malloc`.
 *
 *   WithPoolAllocator::Malloc(100);
 */
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

  /*\brief Allocate memory of size bytes. Returns nullptr if fail. */
  static void *Malloc(size_t size) noexcept;
  static void Free(void* ptr) noexcept;

  // You can define nested context without issue, there's only 1 global pool memory
  // allocator, and the memory won't be freed until the last context goes out of scope.
  int32_t FrameDepth() const { return Context(); }

  bool InContext() const {
    return Context();
  }
};
}      // namespace xgboost
#endif  // XGBOOST_POOL_ALLOCATOR_H_
