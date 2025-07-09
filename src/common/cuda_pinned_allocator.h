/**
 * Copyright 2022-2025, XGBoost Contributors
 *
 * @brief cuda pinned allocator for usage with thrust containers
 */
#pragma once

#include <cuda_runtime.h>

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits
#include <memory>   // for unique_ptr
#include <new>      // for bad_array_new_length
#include <omp.h>

#include "common.h"

namespace xgboost::common::cuda_impl {
// \p pinned_allocator is a CUDA-specific host memory allocator
//  that employs \c cudaMallocHost for allocation.
//
// This implementation is ported from the experimental/pinned_allocator
// that Thrust used to provide.
//
//  \see https://en.cppreference.com/w/cpp/memory/allocator
template <typename T>
struct PinnedAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  [[nodiscard]] constexpr size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_array_new_length{};
    }

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocHost(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFreeHost(p)); }  // NOLINT
};

template <typename T>
struct ManagedAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  [[nodiscard]] constexpr size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_array_new_length{};
    }

    pointer result(nullptr);
    dh::safe_cuda(cudaMallocManaged(reinterpret_cast<void**>(&result), cnt * sizeof(value_type)));
    return result;
  }

  void deallocate(pointer p, size_type) { dh::safe_cuda(cudaFree(p)); }  // NOLINT
};

#if defined(_OPENMP)
#else
#endif  // defined(_OPENMP)

/**
 * @brief Wrapper for the OpenMP allocator.
 *
 * Some useful links for reference and introduction:
 *
 * https://www.openmp.org/spec-html/5.0/openmpsu53.html#x78-2400002.11.2
 * https://www.iwomp.org/wp-content/uploads/iwomp-2023-advanced-openmp-tutorial.pdf
 */
class OmpAllocator {
  omp_allocator_handle_t alloc_;

 public:
  OmpAllocator()
      : alloc_{[] {
          omp_memspace_handle_t ms = omp_default_mem_space;
          std::array<omp_alloctrait_t, 3> traits{
              omp_alloctrait_t{omp_atk_partition, omp_atv_nearest},
              omp_alloctrait_t{omp_atk_sync_hint, omp_atv_uncontended},
              omp_alloctrait_t{omp_atk_alignment, alignof(max_align_t)}};
          return omp_init_allocator(ms, traits.size(), traits.data());
        }()} {}
  ~OmpAllocator() { omp_destroy_allocator(alloc_); }

  OmpAllocator(OmpAllocator const&) = delete;
  OmpAllocator& operator=(OmpAllocator const&) = delete;

  [[nodiscard]] void* Allocate(std::size_t n_bytes) {
    return omp_aligned_alloc(alignof(max_align_t), n_bytes, alloc_);
  }
  void Deallocate(void* ptr) { omp_free(ptr, alloc_); }
};

[[nodiscard]] OmpAllocator& GlobalOmpAllocator();

// This is actually a pinned memory allocator in disguise. We utilize HMM or ATS for
// efficient tracked memory allocation.
template <typename T>
struct SamAllocPolicy {
  using pointer = T*;              // NOLINT: The type returned by address() / allocate()
  using const_pointer = const T*;  // NOLINT: The type returned by address()
  using size_type = std::size_t;   // NOLINT: The type used for the size of the allocation
  using value_type = T;            // NOLINT: The type of the elements in the allocator

  [[nodiscard]] constexpr size_type max_size() const {  // NOLINT
    return std::numeric_limits<size_type>::max() / sizeof(value_type);
  }

  [[nodiscard]] pointer allocate(size_type cnt, const_pointer = nullptr) const {  // NOLINT
    if (cnt > this->max_size()) {
      throw std::bad_array_new_length{};
    }

    size_type n_bytes = cnt * sizeof(value_type);
    auto& alloc = GlobalOmpAllocator();
    pointer result = reinterpret_cast<pointer>(alloc.Allocate(n_bytes));
    if (!result) {
      throw std::bad_alloc{};
    }
    dh::safe_cuda(cudaHostRegister(result, n_bytes, cudaHostRegisterDefault));
    return result;
  }

  void deallocate(pointer p, size_type) {  // NOLINT
    dh::safe_cuda(cudaHostUnregister(p));
    GlobalOmpAllocator().Deallocate(p);
  }
};

/**
 * @brief A RAII handle type to the CUDA memory pool.
 */
using MemPoolHdl = std::unique_ptr<cudaMemPool_t, void (*)(cudaMemPool_t*)>;

/**
 * @brief Create a CUDA memory pool for allocating host pinned memory.
 */
[[nodiscard]] MemPoolHdl CreateHostMemPool();

/**
 * @brief C++ wrapper for the CUDA memory pool.
 */
class HostPinnedMemPool {
  MemPoolHdl pool_;

 public:
  HostPinnedMemPool() : pool_{CreateHostMemPool()} {}
  void* AllocateAsync(std::size_t n_bytes, cudaStream_t stream) {
    void* ptr = nullptr;
    dh::safe_cuda(cudaMallocFromPoolAsync(&ptr, n_bytes, *this->pool_, stream));
    return ptr;
  }
  void DeallocateAsync(void* ptr, cudaStream_t stream) {
    dh::safe_cuda(cudaFreeAsync(ptr, stream));
  }
};

template <typename T, template <typename> typename Policy>
class CudaHostAllocatorImpl : public Policy<T> {
 public:
  using typename Policy<T>::value_type;
  using typename Policy<T>::pointer;
  using typename Policy<T>::const_pointer;
  using typename Policy<T>::size_type;

  using reference = value_type&;              // NOLINT: The parameter type for address()
  using const_reference = const value_type&;  // NOLINT: The parameter type for address()

  using difference_type = std::ptrdiff_t;  // NOLINT: The type of the distance between two pointers

  template <typename U>
  struct rebind {                                    // NOLINT
    using other = CudaHostAllocatorImpl<U, Policy>;  // NOLINT: The rebound type
  };

  CudaHostAllocatorImpl() = default;
  ~CudaHostAllocatorImpl() = default;
  CudaHostAllocatorImpl(CudaHostAllocatorImpl const&) = default;

  CudaHostAllocatorImpl& operator=(CudaHostAllocatorImpl const& that) = default;
  CudaHostAllocatorImpl& operator=(CudaHostAllocatorImpl&& that) = default;

  template <typename U>
  CudaHostAllocatorImpl(CudaHostAllocatorImpl<U, Policy> const&) {}  // NOLINT

  pointer address(reference r) { return &r; }              // NOLINT
  const_pointer address(const_reference r) { return &r; }  // NOLINT

  bool operator==(CudaHostAllocatorImpl const&) const { return true; }

  bool operator!=(CudaHostAllocatorImpl const& x) const { return !operator==(x); }
};

template <typename T>
using PinnedAllocator = CudaHostAllocatorImpl<T, PinnedAllocPolicy>;

template <typename T>
using ManagedAllocator = CudaHostAllocatorImpl<T, ManagedAllocPolicy>;

template <typename T>
using SamAllocator = CudaHostAllocatorImpl<T, SamAllocPolicy>;
}  // namespace xgboost::common::cuda_impl
