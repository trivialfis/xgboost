/*!
 * Copyright (c) by XGBoost Contributors 2020
 */
#include <cinttypes>
#include <cstdlib>

#include "xgboost/logging.h"
#include "xgboost/pool_allocator.h"

namespace xgboost {

class CrtAllocator {
 public:
  static void *Malloc(std::size_t size) {
    if (size) {  //  behavior of malloc(0) is implementation defined.
      return std::malloc(size);
    } else {
      return nullptr;  // standardize to returning NULL.
    }
  }
  static void Free(void *ptr) {
    if (ptr) {
      std::free(ptr);
    }
  }
};

class MemoryPoolAllocator {
  using BaseAllocator = CrtAllocator;

 public:
  template <typename T> static constexpr size_t Align(T x) {
    return (((x) + static_cast<size_t>(7u)) & ~static_cast<size_t>(7u));
  }

  //! Constructor with chunkSize.
  /*! \param chunkSize The size of memory chunk. The default is
     kDefaultChunkSize. \param baseAllocator The allocator for allocating memory
     chunks.
  */
  MemoryPoolAllocator() : chunk_capacity_(kDefaultChunkCapacity) {}

  //! Destructor.
  /*! This deallocates all memory chunks, excluding the user-supplied buffer.
   */
  ~MemoryPoolAllocator() {
    Clear();
    delete own_base_allocator_;
  }

  //! Deallocates all memory chunks, excluding the user-supplied buffer.
  void Clear() {
    while (chunk_head_) {
      ChunkHeader *next = chunk_head_->next;
      base_allocator_->Free(chunk_head_);
      chunk_head_ = next;
    }
  }

  //! Computes the total capacity of allocated memory chunks.
  /*! \return total capacity in bytes.
   */
  size_t Capacity() const {
    size_t capacity = 0;
    for (ChunkHeader *c = chunk_head_; c != nullptr; c = c->next) {
      capacity += c->capacity;
    }
    return capacity;
  }

  //! Allocates a memory block. (concept Allocator)
  void *Malloc(size_t size) {
    if (XGBOOST_EXPECT(!size, false)) {
      return nullptr;
    }

    size = Align(size);
    if (chunk_head_ == nullptr ||
        chunk_head_->size + size > chunk_head_->capacity) {
      if (!AddChunk(chunk_capacity_ > size ? chunk_capacity_ : size)) {
        return nullptr;
      }
    }

    void *buffer = reinterpret_cast<char *>(chunk_head_) +
                   Align(sizeof(ChunkHeader)) + chunk_head_->size;
    chunk_head_->size += size;
    return buffer;
  }

  //! Frees a memory block (concept Allocator)
  static void Free(void *ptr) { (void)ptr; }  // Do nothing

 private:
  //! Creates a new chunk.
  /*! \param capacity Capacity of the chunk in bytes.
      \return true if success.
  */
  bool AddChunk(size_t capacity) {
    if (XGBOOST_EXPECT(!base_allocator_, false)) {
      own_base_allocator_ = base_allocator_ = new BaseAllocator();
    }
    if (ChunkHeader *chunk = reinterpret_cast<ChunkHeader *>(
            base_allocator_->Malloc(Align(sizeof(ChunkHeader)) + capacity))) {
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

  //! Chunk header for perpending to each chunk.
  /*! Chunks are stored as a singly linked list.
   */
  struct ChunkHeader {
    size_t capacity;    //!< Capacity of the chunk in bytes (excluding the header
                        //!< itself).
    size_t size;        //!< Current size of allocated memory in bytes.
    ChunkHeader *next;  //!< Next chunk in the linked list.
  };

  ChunkHeader *chunk_head_{nullptr};  //!< Head of the chunk linked-list. Only the head
                            //!< chunk serves allocation.
  size_t chunk_capacity_;   //!< The minimum capacity of chunk when they are
                            //!< allocated.
  BaseAllocator *base_allocator_{
      nullptr};  //!< base allocator for allocating memory chunks.
  BaseAllocator *own_base_allocator_{
      nullptr};  //!< base allocator created by this object.
};

WithPoolAllocator::~WithPoolAllocator() {
  this->Context()--;
  if (this->Context() == 0) {
    ThreadLocalAllocator::Get()->Clear();
  }
}

void *WithPoolAllocator::Malloc(size_t size) {
  if (Context() != 0) {
    return ThreadLocalAllocator::Get()->Malloc(size);
  } else {
    return CrtAllocator::Malloc(size);
  }
}
void WithPoolAllocator::Free(void *ptr) {
  if (Context() != 0) {
    return ThreadLocalAllocator::Get()->Free(ptr);
  } else {
    return CrtAllocator::Free(ptr);
  }
}
}  // namespace xgboost
