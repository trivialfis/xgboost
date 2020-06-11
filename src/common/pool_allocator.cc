/*!
 * Copyright (c) by XGBoost Contributors 2020
 */
#include <cinttypes>
#include <cstdlib>

#include "xgboost/logging.h"
#include "xgboost/pool_allocator.h"

namespace xgboost {

// A thin wrapper over std::malloc and std::free.
class TrunkAllocator {
 public:
  static void *Malloc(std::size_t size) noexcept {
    if (size != 0) {
      return std::malloc(size);
    } else {
      // Make sure when size is 0, return nullptr instead of implementation defined value.
      return nullptr;
    }
  }
  static void Free(void *ptr) noexcept {
    if (ptr) {
      std::free(ptr);
    }
  }
};

/*\brief A pool memory allocator, similar to arena in tvm, or pool memory allocator in
 *       rapidjson.  See: https://en.wikipedia.org/wiki/Region-based_memory_management for
 *       details.
 */
class MemoryPoolAllocator {
  /*\brief Get 8 bytes aligned size. */
  static constexpr size_t Align(size_t x) {
    return (((x) + static_cast<size_t>(7u)) & ~static_cast<size_t>(7u));
  }

 public:
  MemoryPoolAllocator() = default;
  ~MemoryPoolAllocator() noexcept {
    // Only free memory during destruction.
    Clear();
  }

  /*\brief Clear out all memory usage. */
  void Clear() noexcept {
    while (chunk_head_) {
      ListNode *next = chunk_head_->next;
      base_allocator_->Free(chunk_head_);
      chunk_head_ = next;
    }
  }

  size_t Capacity() const noexcept {
    size_t capacity = 0;
    for (ListNode *chunk = chunk_head_; chunk != nullptr; chunk = chunk->next) {
      capacity += chunk->capacity;
    }
    return capacity;
  }
  /*\brief Allocate memory of size bytes. Returns nullptr if fail. */
  void *Malloc(size_t size) noexcept {
    if (!size) {
      return nullptr;
    }

    size = Align(size);
    if (chunk_head_ == nullptr ||
        chunk_head_->size + size > chunk_head_->capacity) {
      if (!CreateNewChunk(chunk_capacity_ > size ? chunk_capacity_ : size)) {
        return nullptr;
      }
    }

    void *buffer = reinterpret_cast<char *>(chunk_head_) +
                   Align(sizeof(ListNode)) + chunk_head_->size;
    chunk_head_->size += size;
    return buffer;
  }

  static void Free(void *ptr) noexcept { (void)ptr; }

 private:
  bool CreateNewChunk(size_t capacity) {
    if (!base_allocator_) {
      base_allocator_.reset(new TrunkAllocator());
    }
    if (ListNode *chunk = reinterpret_cast<ListNode *>(
            base_allocator_->Malloc(Align(sizeof(ListNode)) + capacity))) {
      chunk->capacity = capacity;
      chunk->size = 0;
      chunk->next = chunk_head_;
      chunk_head_ = chunk;
      return true;
    } else {
      return false;
    }
  }

  static const int kDefaultChunkCapacity = (64 * 1024);

  struct ListNode {
    size_t capacity;
    size_t size;
    ListNode *next;
  };

  ListNode *chunk_head_{nullptr};
  size_t chunk_capacity_ {kDefaultChunkCapacity};

  std::unique_ptr<TrunkAllocator> base_allocator_;
};

WithPoolAllocator::~WithPoolAllocator() {
  this->Context()--;
  if (this->Context() == 0) {
    ThreadLocalAllocator::Get()->Clear();
  }
}

void *WithPoolAllocator::Malloc(size_t size) noexcept {
  if (Context() != 0) {
    return ThreadLocalAllocator::Get()->Malloc(size);
  } else {
    return TrunkAllocator::Malloc(size);
  }
}

void WithPoolAllocator::Free(void *ptr) noexcept {
  if (Context() != 0) {
    return ThreadLocalAllocator::Get()->Free(ptr);
  } else {
    return TrunkAllocator::Free(ptr);
  }
}
}  // namespace xgboost
