/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <cstdint>               // uint32_t, int32_t
#include <cuda/std/type_traits>  // for cuda::std::alignment_of_v
#include <memory>                // for unique_ptr
#include <utility>               // for pair

#include "../../collective/aggregator.h"
#include "../../common/cuda_compat.cuh"   // for CUDA compatibility
#include "../../common/cuda_context.cuh"  // for CUDAContext
#include "../../common/cuda_rt_utils.h"   // for GetMpCnt
#include "../../common/device_helpers.cuh"
#include "../../data/ellpack_page.cuh"
#include "histogram.cuh"
#include "row_partitioner.cuh"
#include "xgboost/base.h"

namespace xgboost::tree {
namespace {
template <typename IterT>
XGBOOST_DEV_INLINE bst_idx_t IterIdx(EllpackAccessorImpl<IterT> const& matrix,
                                     RowPartitioner::RowIndexT ridx, bst_feature_t fidx) {
  // # Row index local to each batch
  // ridx_local = ridx - base_rowid
  // # Starting entry index for this row in the matrix
  // entry_idx = ridx_local * row_stride
  // # Inside a row, first column inside this feature group
  // entry_idx += start_feature
  // # The feature index local to the current feature group
  // idx - ridx * feature_stride == idx % feature_stride
  // # Final index
  // entry_idx += idx % feature_stride
  return (ridx - matrix.base_rowid) * matrix.row_stride + fidx;
}
}  // anonymous namespace

XGBOOST_DEV_INLINE void AtomicAddGpairShared(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<int64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  AtomicAdd64As32(dst_ptr, g);
  AtomicAdd64As32(dst_ptr + 1, h);
}

// Global 64 bit integer atomics at the time of writing do not benefit from being separated into two
// 32 bit atomics
XGBOOST_DEV_INLINE void AtomicAddGpairGlobal(xgboost::GradientPairInt64* dest,
                                             xgboost::GradientPairInt64 const& gpair) {
  auto dst_ptr = reinterpret_cast<uint64_t*>(dest);
  auto g = gpair.GetQuantisedGrad();
  auto h = gpair.GetQuantisedHess();

  atomicAdd(dst_ptr, *reinterpret_cast<uint64_t*>(&g));
  atomicAdd(dst_ptr + 1, *reinterpret_cast<uint64_t*>(&h));
}

template <std::int32_t BlockThreads, std::int32_t MinBlocks>
struct HistTuning {
  static constexpr std::int32_t kBlockThreads = BlockThreads;
  static constexpr std::int32_t kMinBlocks = MinBlocks;
};

namespace {
constexpr std::int32_t kItemsPerThread = 8;

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#feature-set-compiler-targets
// Technical Specifications                  7.5  | 8.0  | 8.6  8.7 | 8.9 | 9.0 10.0 | 11.0 12.0
// Maximum number of resident blocks per SM  16   | 32   | 16       | 24  | 32       | 24
// Maximum number of resident warps per SM   32   | 64   | 48             | 64       | 48
// Maximum number of resident threads per SM 1024 | 2048 | 1536           | 2048     | 1536

using HistSm75 = HistTuning<1024, 1>;

using HistSm80 = HistTuning<1024, 2>;

using HistSm86 = HistTuning<768, 2>;

using HistSm90 = HistTuning<1024, 2>;

using HistSm110 = HistTuning<768, 2>;

// Multi-target launch bounds
#if __CUDA_ARCH__ >= 1100
using MtHistBound = HistSm110;
#elif __CUDA_ARCH__ >= 900
using MtHistBound = HistSm90;
#elif __CUDA_ARCH__ >= 860
using MtHistBound = HistSm86;
#elif __CUDA_ARCH__ >= 800
using MtHistBound = HistSm80;
#else
using MtHistBound = HistSm75;
#endif

// Single-target launch bounds
// Maximize the number of threads instead of tuning for occupancy for single target. The
// histogram uses the largest shared memory possible, which limits the occupancy on most
// archs. Archs with 2048 threads per SM can still fit two blocks if the kernel uses at
// most 32 registers.
struct StHistBound {
  static constexpr std::int32_t kBlockThreads = 1024;
};
// The multi-target tuning is for full occupancy.
constexpr std::int32_t kMaxThreadsPerSm = MtHistBound::kBlockThreads * MtHistBound::kMinBlocks;
using StHistDeviceBound = HistTuning<StHistBound::kBlockThreads, kMaxThreadsPerSm / 1024>;

template <typename HistArchPolicy, std::int32_t ItemsPerThread, bool Dense, bool Compressed,
          bool SharedMem>
struct HistPolicy : public HistArchPolicy {
  using ArchPolicy = HistArchPolicy;
  static constexpr std::int32_t kItemsPerThread = ItemsPerThread;
  static constexpr std::int32_t kTileSize = HistArchPolicy::kBlockThreads * ItemsPerThread;
  static constexpr bool kDense = Dense;
  static constexpr bool kCompressed = Compressed;
  static constexpr bool kSharedMem = SharedMem;
  // The cost of zeroing and flushing the privatized histogram for a segment, in items.
  static constexpr std::int32_t kSegmentCost = SharedMem ? kTileSize : 0;
  static constexpr bool kSingleTarget = std::is_same_v<HistArchPolicy, StHistBound>;
};

// The launch bounds depend on `__CUDA_ARCH__`, they must be resolved in the device
// compilation pass instead of being used as template arguments.
template <typename Policy>
using HistBound = std::conditional_t<Policy::kSingleTarget, StHistDeviceBound, MtHistBound>;

template <typename Fn>
void DispatchCudaSm(std::int32_t device, Fn&& fn) {
  std::int32_t version = 0;
  dh::safe_cuda(cub::SmVersion(version, device));
  if (version >= 1100) {
    fn(HistSm110{});
  } else if (version >= 900) {
    fn(HistSm90{});
  } else if (version >= 860) {
    fn(HistSm86{});
  } else if (version >= 800) {
    fn(HistSm80{});
  } else {
    fn(HistSm75{});
  }
}

__device__ GradientPairInt64 LoadGpair(GradientPairInt64 const* XGBOOST_RESTRICT gpairs) {
  static_assert(sizeof(int4) == sizeof(GradientPairInt64));
  auto g = *reinterpret_cast<int4 const*>(gpairs);
  return *reinterpret_cast<GradientPairInt64*>(&g);
}

// Build the histogram for the items [begin, end) of a single node, target, and feature group.
template <typename Policy, typename Accessor, typename RidxIterSpan>
__device__ void HistKernelOneNodeTarget(Accessor const& matrix, FeatureGroup const& group,
                                        RidxIterSpan d_ridx_iter, GradientPairInt64 const* gpair,
                                        GradientPairInt64* smem_hist, GradientPairInt64* gmem_hist,
                                        bst_idx_t begin, bst_idx_t end) {
  bst_feature_t const feature_stride = Policy::kCompressed ? group.num_features : matrix.row_stride;

  using Idx = RowPartitioner::RowIndexT;

  auto const d_ridx = d_ridx_iter.data();

  auto atomic_add = [&](auto bin_idx, auto const& adjusted) {
    if constexpr (Policy::kSharedMem) {
      AtomicAddGpairShared(smem_hist + bin_idx, adjusted);
    } else {
      // gmem_hist is a subspan for the current target.
      AtomicAddGpairGlobal(gmem_hist + bin_idx, adjusted);
    }
  };

  auto process_valid_tile = [&](auto idx) {
    // unrolled version unravel to save registers:
    // auto [ridx, fidx] = unravel_index(idx, (n_rows, feature_stride));
    //
    // ridx_in_set: Index into the row batch
    // fidx_in_set: Index into the feature group
    Idx ridx_in_set = idx / feature_stride;
    Idx fidx_in_set = idx - ridx_in_set * feature_stride;

    Idx ridx = d_ridx[ridx_in_set];
    auto fidx = fidx_in_set + group.start_feature;

    bst_bin_t compressed_bin = matrix.gidx_iter[IterIdx(matrix, ridx, fidx)];
    if (Policy::kDense || compressed_bin != static_cast<bst_bin_t>(matrix.NullValue())) {
      auto g = LoadGpair(gpair + ridx);
      if constexpr (Policy::kCompressed) {
        compressed_bin += matrix.feature_segments[fidx];
      }
      if constexpr (Policy::kSharedMem) {
        compressed_bin -= group.start_bin;
      }
      atomic_add(compressed_bin, g);
    }
  };

  auto process_gpair_tile = [&](auto full_tile, auto offset) {
#pragma unroll 1
    for (std::int32_t j = 0; j < Policy::kItemsPerThread; ++j) {
      bst_idx_t const idx = offset + j * Policy::kBlockThreads + threadIdx.x;
      if (full_tile || idx < end) {
        process_valid_tile(idx);
      }
    }
  };

  for (auto offset = begin; offset < end; offset += Policy::kTileSize) {
    if (end - offset >= static_cast<bst_idx_t>(Policy::kTileSize)) {
      process_gpair_tile(std::true_type{}, offset);
    } else {
      process_gpair_tile(std::false_type{}, offset);
    }
  }
}

// A range of items inside a (node, feature group, target) segment.
struct HistSegment {
  std::size_t nidx_in_set;
  bst_target_t target_idx;
  bst_feature_t gidx;
  // The range of valid items local to the segment, empty if the position is in the padding.
  bst_idx_t begin;
  bst_idx_t end;
  // The distance to the next segment or to `last`, whichever is closer.
  bst_idx_t step;
};

// The largest index `i` in [0, n) with `begin(i) <= pos`, `begin` must be non-decreasing.
template <typename Fn>
XGBOOST_DEV_INLINE std::size_t UpperBoundIdx(std::size_t n, bst_idx_t pos, Fn&& begin) {
  std::size_t base = 0;
  while (n > 1) {
    auto half = n / 2;
    base = begin(base + half) <= pos ? base + half : base;
    n -= half;
  }
  return base;
}

// Find the segment of the item at `pos`, and the range of items in this segment up to
// `last`. Each segment is followed by `Policy::kSegmentCost` padding items to account for
// the fixed cost of the segment when slicing items for blocks. The padding contains no
// valid item.
template <typename Policy, typename Accessor>
XGBOOST_DEV_INLINE HistSegment FindSegment(Accessor const& matrix,
                                           FeatureGroupsAccessor const& feature_groups,
                                           common::Span<std::size_t const> sizes_csum,
                                           bst_target_t n_targets, bst_idx_t pos, bst_idx_t last) {
  constexpr bst_idx_t kSegCost = Policy::kSegmentCost;
  auto const n_groups = feature_groups.NumGroups();
  auto const* XGBOOST_RESTRICT p_sizes = sizes_csum.data();
  // Number of items in a row for each target. Without compression, each group scans the
  // entire row.
  bst_idx_t const row_items =
      Policy::kCompressed ? matrix.row_stride : matrix.row_stride * n_groups;

  HistSegment seg;
  seg.nidx_in_set = UpperBoundIdx(sizes_csum.size() - 1, pos, [&](std::size_t i) {
    return n_targets * (p_sizes[i] * row_items + i * n_groups * kSegCost);
  });
  auto nidx = seg.nidx_in_set;
  bst_idx_t const n_rows = p_sizes[nidx + 1] - p_sizes[nidx];
  bst_idx_t offset = pos - n_targets * (p_sizes[nidx] * row_items + nidx * n_groups * kSegCost);
  // Inside a node, all targets of a feature group are next to each other.
  bst_idx_t group_size;
  if constexpr (Policy::kCompressed) {
    auto const* XGBOOST_RESTRICT p_fs = feature_groups.feature_segments.data();
    auto group_begin = [&](bst_feature_t g) {
      return n_targets * (n_rows * p_fs[g] + g * kSegCost);
    };
    seg.gidx = UpperBoundIdx(n_groups, offset, group_begin);
    offset -= group_begin(seg.gidx);
    group_size = p_fs[seg.gidx + 1] - p_fs[seg.gidx];
  } else {
    group_size = matrix.row_stride;
    seg.gidx = offset / (n_targets * (n_rows * group_size + kSegCost));
    offset -= seg.gidx * n_targets * (n_rows * group_size + kSegCost);
  }
  bst_idx_t const n_valid = n_rows * group_size;
  seg.target_idx = 0;
  if (n_targets > 1) {
    seg.target_idx = offset / (n_valid + kSegCost);
    offset -= seg.target_idx * (n_valid + kSegCost);
  }
  seg.begin = offset;
  seg.end = cuda::std::min(n_valid, seg.begin + (last - pos));
  seg.step = cuda::std::min(n_valid + kSegCost - offset, last - pos);
  return seg;
}
}  // namespace

/**
 * @brief Kernel for building histograms of multiple nodes and targets.
 *
 * @param matrix          An ellpack accessor.
 * @param feature_groups  Grouping for privatized histogram.
 * @param d_ridx_iters    Pointer to row index spans. One span per node.
 * @param sizes_csum      Cumulative sum of the number of rows in each node.
 * @param node_hists      Pointer to histograms. One histogram per node.
 * @param items_per_block The number of items processed by each block.
 * @param n_items         The total number of items.
 *
 * The items of all nodes, feature groups, and targets are concatenated in this order, and
 * each block processes a contiguous range of items. The range of a block can span
 * multiple segments of (node, group, target); the block flushes its privatized histogram
 * once for each segment. As a result, the number of flushes is bounded by the number of
 * blocks plus the number of segments. Targets are the innermost dimension so that blocks
 * of different targets read the same bin indices around the same time, sharing them in L2.
 *
 * Each segment is padded with the cost of its flush. Otherwise, a block can receive many
 * small segments (small nodes) and flush them sequentially while other blocks are idle.
 */
template <typename Policy, typename Accessor, typename RidxIterSpan>
__global__ __launch_bounds__(
    HistBound<Policy>::kBlockThreads,
    HistBound<Policy>::kMinBlocks) void HistogramKernel(Accessor const matrix,
                                                        FeatureGroupsAccessor const feature_groups,
                                                        RidxIterSpan const* d_ridx_iters,
                                                        common::Span<std::size_t const> sizes_csum,
                                                        common::Span<GradientPairInt64> const*
                                                            node_hists,
                                                        GradientPairInt64 const* d_gpair,
                                                        bst_idx_t n_samples, bst_target_t n_targets,
                                                        bst_idx_t items_per_block,
                                                        bst_idx_t n_items) {
  if constexpr (Policy::kSingleTarget) {
    // Constant propagation removes the target indexing and saves registers.
    n_targets = 1;
  }

  extern __align__(std::alignment_of_v<GradientPairInt64>) __shared__ char shmem[];
  // Privatized histogram
  auto smem_hist = reinterpret_cast<GradientPairInt64*>(shmem);

  auto find_segment = [&](bst_idx_t pos) {
    // The end of the range for this block. Derived from the position instead of the block
    // index to avoid keeping it alive.
    bst_idx_t last = cuda::std::min(pos - pos % items_per_block + items_per_block, n_items);
    return FindSegment<Policy>(matrix, feature_groups, sizes_csum, n_targets, pos, last);
  };
  auto target_hist = [&](HistSegment const& seg) {
    auto d_node_hist = node_hists[seg.nidx_in_set];
    // With a target-major layout, we don't have to pack the histogram for all targets into
    // the shared memory.
    auto gmem_hist = d_node_hist.data() + seg.target_idx * (d_node_hist.size() / n_targets);
    // The pointer is loaded from global memory, without the hint, the compiler emits generic
    // atomics for the flush.
    __builtin_assume(__isGlobal(gmem_hist));
    return gmem_hist;
  };

  // The position is the only state carried between segments. Ranges of blocks are aligned
  // to `items_per_block`, the block reaches the end of its range when the position is
  // aligned.
  bst_idx_t pos = blockIdx.x * items_per_block;
  do {
    auto seg = find_segment(pos);
    if (seg.begin < seg.end) {
      auto group = feature_groups[seg.gidx];
      if constexpr (Policy::kSharedMem) {
        // Each thread zeroes the same bins it flushes, no barrier is needed after the flush
        // of the previous segment.
        dh::BlockFill(smem_hist, group.num_bins, GradientPairInt64{});
        __syncthreads();
      }
      HistKernelOneNodeTarget<Policy>(matrix, group, d_ridx_iters[seg.nidx_in_set],
                                      d_gpair + n_samples * seg.target_idx, smem_hist,
                                      target_hist(seg), seg.begin, seg.end);
      if constexpr (Policy::kSharedMem) {
        __syncthreads();
        // Recompute the segment instead of keeping it alive across the histogram loop,
        // which saves registers.
        asm volatile("" : "+l"(pos));
        seg = find_segment(pos);
        group = feature_groups[seg.gidx];
        auto gmem_hist = target_hist(seg);
        // Write shared memory back to global memory
        for (auto bin_idx : dh::BlockStrideRange(0, group.num_bins)) {
          AtomicAddGpairGlobal(gmem_hist + group.start_bin + bin_idx, smem_hist[bin_idx]);
        }
      }
    }
    pos += seg.step;
  } while (pos < n_items && pos % items_per_block != 0);
}

// Dispatcher for the histogram kernel.
struct HistKernel {
  /**
   * @brief Split the items into equal ranges of whole tiles, one range for each block.
   *
   * Each block zeroes and flushes the privatized histogram at least once, which is
   * comparable to processing a tile. A block needs enough tiles to amortize this fixed
   * cost. On the other hand, multiple waves of blocks balance the load between SMs. For
   * small inputs, filling the device takes priority over amortizing the flush.
   *
   * @param n_items           The total number of items, including the segment padding.
   * @param n_resident_blocks The number of blocks that the device can run concurrently.
   *
   * @return The number of items for each block and the number of blocks.
   */
  template <typename Policy>
  static auto SliceItems(bst_idx_t n_items, std::size_t n_resident_blocks) {
    CHECK_GT(n_resident_blocks, 0);
    constexpr std::size_t kMaxWaves = 32;
    constexpr std::size_t kMinTiles = 32;
    auto n_tiles = common::DivRoundUp(n_items, Policy::kTileSize);
    auto min_tiles = std::min(kMinTiles, common::DivRoundUp(n_tiles, n_resident_blocks));
    auto tiles_per_block =
        std::max(common::DivRoundUp(n_tiles, n_resident_blocks * kMaxWaves), min_tiles);
    auto n_blocks = common::DivRoundUp(n_tiles, tiles_per_block);
    CHECK_LE(n_blocks, std::numeric_limits<std::uint32_t>::max());
    return std::make_pair(static_cast<bst_idx_t>(tiles_per_block * Policy::kTileSize),
                          static_cast<std::uint32_t>(n_blocks));
  }

  struct HistKernelConfig {
    std::int32_t n_blocks_per_mp = 0;
    std::size_t shmem_bytes = 0;

    template <typename Policy, typename Kernel>
    void Reset(std::size_t new_shmem_bytes, Kernel* kernel, Policy, std::size_t max_shared_bytes) {
      if (new_shmem_bytes > 0) {
        // This function is the reason for all this trouble to cache the
        // configuration. It blocks the device.
        //
        // Also, it must precede the `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
        // otherwise the shmem bytes might be invalid.
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           max_shared_bytes));
      }
      if (new_shmem_bytes > this->shmem_bytes) {
        this->shmem_bytes = new_shmem_bytes;
      }
      // Use this as a limiter, works for root node. Not too bad an option for child nodes.
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &this->n_blocks_per_mp, kernel, Policy::kBlockThreads, shmem_bytes));
    }
  };

  // Maps kernel instantiations to their configurations. This is a mutable state, as a
  // result the histogram kernel is not thread safe.
  std::map<void*, HistKernelConfig> cfg;
  // The number of multi-processor for the selected GPU
  std::int32_t const n_mps;
  // Maximum size of the shared memory (optin)
  std::size_t const max_shared_bytes;
  // Use global memory for testing
  bool const force_global;

  template <typename Policy, typename Kernel>
  void SetCfg(Policy policy, std::size_t shmem_bytes, Kernel kernel) {
    auto it = this->cfg.find(reinterpret_cast<void*>(kernel));

    HistKernelConfig v;
    if (it == cfg.cend()) {
      v.Reset(shmem_bytes, kernel, policy, max_shared_bytes);
      this->cfg[reinterpret_cast<void*>(kernel)] = v;
    }
  }

  explicit HistKernel(Context const* ctx, bool force_global)
      : n_mps{curt::GetMpCnt(ctx->Ordinal())},
        max_shared_bytes{dh::MaxSharedMemoryOptin(ctx->Ordinal())},
        force_global{force_global} {}

  template <bool kDense, bool kCompressed, typename Accessor, typename RidxIterSpan>
  void DispatchHistShmem(Context const* ctx, Accessor const& matrix,
                         FeatureGroupsAccessor const& feature_groups,
                         linalg::MatrixView<GradientPairInt64 const> gpair,
                         RidxIterSpan* ridx_iters,
                         common::Span<common::Span<GradientPairInt64>> hists,
                         std::vector<std::size_t> const& h_sizes_csum) {
    CHECK(gpair.FContiguous());
    auto n_samples = gpair.Shape(0);
    auto n_targets = gpair.Shape(1);
    auto d_gpair = gpair.Values().data();

    std::size_t shmem_bytes = feature_groups.ShmemSize();
    bool use_shared = !force_global && shmem_bytes <= this->max_shared_bytes;
    shmem_bytes = use_shared ? shmem_bytes : 0;

    auto launch = [&](auto policy, auto kernel) {
      auto const& v = this->cfg.at(reinterpret_cast<void*>(kernel));
      using Policy = common::GetValueT<decltype(policy)>;
      CHECK_GT(v.n_blocks_per_mp, 0);
      if (h_sizes_csum.back() == 0) {
        return;
      }
      // Must match the kernel.
      bst_idx_t row_items =
          Policy::kCompressed ? matrix.row_stride : matrix.row_stride * feature_groups.NumGroups();
      bst_idx_t n_segments = (h_sizes_csum.size() - 1) * feature_groups.NumGroups();
      bst_idx_t n_items =
          n_targets * (h_sizes_csum.back() * row_items + n_segments * Policy::kSegmentCost);
      auto [items_per_block, n_blocks] = SliceItems<Policy>(n_items, v.n_blocks_per_mp * n_mps);
      dh::device_vector<std::size_t> sizes_csum{h_sizes_csum};
      dh::LaunchKernel(n_blocks, Policy::kBlockThreads, shmem_bytes, ctx->CUDACtx()->Stream())(
          kernel, matrix, feature_groups, ridx_iters, dh::ToSpan(sizes_csum), hists.data(), d_gpair,
          n_samples, n_targets, items_per_block, n_items);
      dh::safe_cuda(cudaPeekAtLastError());
    };

    // Single target maximizes the number of threads, multi-target tunes for occupancy.
    auto dispatch_arch = [&](auto&& fn) {
      if (n_targets == 1) {
        fn(StHistBound{});
      } else {
        DispatchCudaSm(ctx->Ordinal(), fn);
      }
    };
    if (use_shared) {
      dispatch_arch([&](auto arch) {
        using Arch = common::GetValueT<decltype(arch)>;
        using Policy = HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, true>;
        auto kernel = HistogramKernel<Policy, Accessor, RidxIterSpan>;
        this->SetCfg(Policy{}, shmem_bytes, kernel);
        launch(Policy{}, kernel);
      });
    } else {
      dispatch_arch([&](auto arch) {
        using Arch = common::GetValueT<decltype(arch)>;
        using Policy = HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, false>;
        auto kernel = HistogramKernel<Policy, Accessor, RidxIterSpan>;
        this->SetCfg(Policy{}, shmem_bytes, kernel);
        launch(Policy{}, kernel);
      });
    }
  }

  template <typename Accessor, typename... Args>
  void DispatchHistCompress(Context const* ctx, Accessor const& matrix, Args&&... args) {
    if (matrix.IsDense()) {
      DispatchHistShmem<true, true>(ctx, matrix, std::forward<Args>(args)...);
    } else if (matrix.IsDenseCompressed()) {
      DispatchHistShmem<false, true>(ctx, matrix, std::forward<Args>(args)...);
    } else {
      DispatchHistShmem<false, false>(ctx, matrix, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  void Dispatch(Args&&... args) {
    this->DispatchHistCompress(std::forward<Args>(args)...);
  }
};

template <typename Accessor>
class DeviceHistogramDispatchAccessor {
  std::unique_ptr<HistKernel> kernel_{nullptr};

 public:
  void Reset(Context const* ctx, bool force_global_memory) {
    this->kernel_ = std::make_unique<HistKernel>(ctx, force_global_memory);
  }

  void BuildHistogram(Context const* ctx, Accessor const& matrix,
                      FeatureGroupsAccessor const& feature_groups,
                      linalg::MatrixView<GradientPairInt64 const> gpair,
                      common::Span<common::Span<cuda_impl::RowIndexT const>> ridxs,
                      common::Span<common::Span<GradientPairInt64>> hists,
                      std::vector<std::size_t> const& h_sizes_csum) {
    std::size_t n_total_samples = h_sizes_csum.back();
    if (ridxs.size() == 1 && n_total_samples == matrix.n_rows) {
      // Special optimization for the root node.
      using RidxIter = dh::counting_iterator<cuda_impl::RowIndexT>;
      CHECK_LT(matrix.base_rowid, std::numeric_limits<cuda_impl::RowIndexT>::max());
      auto iter = common::IterSpan{
          dh::make_counting_iterator(static_cast<cuda_impl::RowIndexT>(matrix.base_rowid)),
          matrix.n_rows};
      dh::caching_device_vector<common::IterSpan<RidxIter>> ridx_iters(hists.size(), iter);
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridx_iters.data().get(), hists,
                              h_sizes_csum);
    } else {
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridxs.data(), hists,
                              h_sizes_csum);
    }
  }
};

// Dispatch between single buffer accessor and double buffer accessor.
struct DeviceHistogramBuilderImpl {
  DeviceHistogramDispatchAccessor<EllpackDeviceAccessor> simpl;
  DeviceHistogramDispatchAccessor<DoubleEllpackAccessor> dimpl;

  template <typename... Args>
  void Reset(Args&&... args) {
    this->simpl.Reset(std::forward<Args>(args)...);
    this->dimpl.Reset(std::forward<Args>(args)...);
  }

  template <typename Accessor, typename... Args>
  void BuildHistogram(Context const* ctx, Accessor const& matrix, Args&&... args) {
    if constexpr (std::is_same_v<Accessor, EllpackDeviceAccessor>) {
      this->simpl.BuildHistogram(ctx, matrix, std::forward<Args>(args)...);
    } else {
      static_assert(std::is_same_v<Accessor, DoubleEllpackAccessor>);
      this->dimpl.BuildHistogram(ctx, matrix, std::forward<Args>(args)...);
    }
  }
};

DeviceHistogramBuilder::DeviceHistogramBuilder()
    : p_impl_{std::make_unique<DeviceHistogramBuilderImpl>()} {
  monitor_.Init(__func__);
}

DeviceHistogramBuilder::~DeviceHistogramBuilder() = default;

void DeviceHistogramBuilder::Reset(Context const* ctx, std::size_t max_cached_hist_nodes,
                                   bst_bin_t n_total_bins, bool force_global_memory) {
  this->monitor_.Start(__func__);
  this->p_impl_->Reset(ctx, force_global_memory);
  this->hist_.Reset(ctx, n_total_bins, max_cached_hist_nodes);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::BuildHistogram(Context const* ctx, EllpackAccessor const& matrix,
                                            FeatureGroupsAccessor const& feature_groups,
                                            common::Span<GradientPairInt64 const> gpair,
                                            common::Span<cuda_impl::RowIndexT const> ridx,
                                            common::Span<GradientPairInt64> histogram) {
  if (ridx.empty()) {
    return;
  }
  dh::caching_device_vector<common::Span<cuda_impl::RowIndexT const>> ridxs(1, ridx);
  dh::caching_device_vector<common::Span<GradientPairInt64>> hists(1, histogram);
  this->BuildHistogram(ctx, matrix, feature_groups,
                       linalg::MakeTensorView(ctx, linalg::kF, gpair, gpair.size(), 1),
                       dh::ToSpan(ridxs), dh::ToSpan(hists), {0, ridx.size()});
}

void DeviceHistogramBuilder::BuildHistogram(
    Context const* ctx, EllpackAccessor const& matrix, FeatureGroupsAccessor const& feature_groups,
    linalg::MatrixView<GradientPairInt64 const> gpair,
    common::Span<common::Span<cuda_impl::RowIndexT const>> ridxs,
    common::Span<common::Span<GradientPairInt64>> hists,
    std::vector<std::size_t> const& h_sizes_csum) {
  this->monitor_.Start(__func__);
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridxs, hists,
                                      h_sizes_csum);
      },
      matrix);
  this->monitor_.Stop(__func__);
}

void DeviceHistogramBuilder::AllReduceHist(Context const* ctx, bst_node_t nidx,
                                           std::size_t num_histograms) {
  this->monitor_.Start(__func__);
  auto d_node_hist = hist_.GetNodeHistogram(nidx);
  using ReduceT = typename std::remove_pointer_t<decltype(d_node_hist.data())>::ValueT;
  auto rc = collective::GlobalSum(
      ctx, linalg::MakeVec(reinterpret_cast<ReduceT*>(d_node_hist.data()),
                           d_node_hist.size() * 2 * num_histograms, ctx->Device()));
  SafeColl(rc);
  this->monitor_.Stop(__func__);
}
}  // namespace xgboost::tree
