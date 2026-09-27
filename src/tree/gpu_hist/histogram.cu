/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <algorithm>             // for min, max
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
  // # Final index, `fidx` is the column in the row, the caller resolves it from the index
  // # local to the feature group.
  // entry_idx += fidx
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

// For reference only, the shared memory budget is derived from the device by
// `DftStHistShmemBytes` and the occupancy by `HistKernel::BlocksPerMp`.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#feature-set-compiler-targets
// Technical Specifications                  7.5  | 8.0  | 8.6  8.7 | 8.9 | 9.0 10.0 | 11.0 12.0
// Maximum number of resident blocks per SM  16   | 32   | 16       | 24  | 32       | 24
// Maximum number of resident warps per SM   32   | 64   | 48             | 64       | 48
// Maximum number of resident threads per SM 1024 | 2048 | 1536           | 2048     | 1536
// Maximum shared memory per SM (KB)         64   | 164  | 100  164 | 100 | 228      | 100

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
// Maximize the number of threads instead of tuning for occupancy for single target, so the
// block size is fixed. Only the tag type is needed on the host, the co-residency is
// resolved in the device pass below.
struct StHistBound {
  static constexpr std::int32_t kBlockThreads = 1024;
};

// `__launch_bounds__` caps the registers per thread at
// `regs_per_sm / (block_threads * min_blocks)`, so `min_blocks` is the number of blocks we
// want co-resident on an SM. Asking for more than can actually fit does not gain occupancy,
// it only tightens the register cap and spills.
//
// Fill the SM: as many blocks as its thread budget allows. `DftStHistShmemBytes` then splits
// the shared memory of the SM between exactly this many blocks, so the shared memory can
// never be the reason a block fails to be co-resident. Both derive from the thread budget of
// the SM, hence they can not disagree. The multi-target tuning already targets full
// occupancy, so its thread budget is the same quantity.
constexpr std::int32_t kStMinBlocks =
    std::max(1, MtHistBound::kBlockThreads * MtHistBound::kMinBlocks /
                    StHistBound::kBlockThreads);
using StHistDeviceBound = HistTuning<StHistBound::kBlockThreads, kStMinBlocks>;

template <typename HistArchPolicy, std::int32_t ItemsPerThread, bool Dense, bool Compressed,
          bool SharedMem>
struct HistPolicy : public HistArchPolicy {
  static constexpr std::int32_t kItemsPerThread = ItemsPerThread;
  // The scheduling granularity. Items are no longer processed in tiles by the accumulation
  // loop, the tile is only the smallest unit of work handed to a block and the unit for
  // accounting the cost of a flush.
  static constexpr std::int32_t kTileSize = HistArchPolicy::kBlockThreads * kItemsPerThread;
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
using HistBound =
    std::conditional_t<Policy::kSingleTarget, StHistDeviceBound, MtHistBound>;

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

// Upper bound on the shared memory a histogram block may request.
//
// Without it, the derivation below gives 113KB on sm_90 and sm_100, where the previous
// per-arch heuristic gave 96KB, and a small regression was reported on H200 after that
// change. Every other supported arch derives less than this, so the cap only affects those
// two, i.e. it restores exactly the budget they had before.
//
// FIXME(jiamingy): The reason a larger budget hurts there is not understood. It is not L1
// capacity: shared memory and L1 share one unified data cache, and taking 113KB x 2 blocks
// forces the maximum carve-out, which leaves L1 at its 28KB minimum instead of 60KB. That
// mechanism is real and was measured on sm_120 (a synthetic kernel with a 48KB working set is
// 8.7x slower under the large carve-out), but the histogram kernel does not depend on L1
// capacity: it streams `gidx`, and the data it actually reuses (`d_ridx`, the segment arrays)
// is only a few KB. Measured on sm_120 with interleaved pairs, shrinking the budget to keep
// L1 was neutral for single target (ratio 0.985-1.004) and a consistent small loss for multi
// target (1.015-1.033), so that is not the fix. Raise or drop this cap once someone can
// profile an affected device.
constexpr std::size_t kMaxShmemBytes = 96 * 1024;

/**
 * @brief The shared memory budget for a block, given the co-residency the launch bounds
 *        ask for.
 *
 * `__launch_bounds__(block_threads, min_blocks)` promises that `min_blocks` blocks can be
 * co-resident, and ptxas caps the registers per thread to make it so. Shared memory must not
 * then be the reason the promise can not be kept, so the shared memory of the SM is split
 * between exactly `min_blocks` blocks.
 *
 * Using less leaves shared memory unused and forces `FeatureGroups` to create more groups
 * than necessary. Using more costs a co-resident block, which the launch bounds already paid
 * for with a tighter register cap.
 *
 * Note that the aggregate shared memory used per SM is the full amount for any `min_blocks`,
 * only the per-block share changes.
 *
 * The budget is capped at `kMaxShmemBytes`, see there.
 */
[[nodiscard]] std::size_t HistShmemBytes(std::int32_t device, std::int32_t min_blocks) {
  CHECK_GT(min_blocks, 0);
  auto optin = dh::MaxSharedMemoryOptin(device);
  std::int32_t smem_per_sm = 0, reserved = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&smem_per_sm,
                                       cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  dh::safe_cuda(
      cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device));

  // Each block is additionally charged a fixed driver reservation. Round down to the
  // allocation granularity, otherwise the last block does not fit.
  constexpr std::int32_t kGranularity = 128;
  auto per_block = (smem_per_sm / min_blocks / kGranularity) * kGranularity - reserved;
  CHECK_GT(per_block, 0);
  // A block can not request more than the opt-in maximum.
  return std::min({static_cast<std::size_t>(per_block), optin, kMaxShmemBytes});
}

// The co-residency the launch bounds ask for, as seen from the host. The device pass uses
// `HistBound<Policy>::kMinBlocks`; these must agree, which is why both are derived from the
// thread budget of the SM.
[[nodiscard]] std::int32_t HistMinBlocks(std::int32_t device, std::int32_t block_threads) {
  std::int32_t max_threads_per_sm = 0;
  dh::safe_cuda(cudaDeviceGetAttribute(&max_threads_per_sm,
                                       cudaDevAttrMaxThreadsPerMultiProcessor, device));
  return std::max(1, max_threads_per_sm / block_threads);
}
}  // anonymous namespace

std::size_t DftStHistShmemBytes(std::int32_t device) {
  // Single target uses a fixed block size, see `StHistBound`.
  return HistShmemBytes(device, HistMinBlocks(device, StHistBound::kBlockThreads));
}

std::size_t DftMtHistShmemBytes(std::int32_t device) {
  // Multi target picks the block size per arch, so the co-residency comes from the tuning
  // rather than from the thread budget. `DispatchCudaSm` resolves the same tag the device
  // pass resolves through `__CUDA_ARCH__`.
  std::size_t bytes = 0;
  DispatchCudaSm(device, [&](auto arch) {
    using Arch = common::GetValueT<decltype(arch)>;
    // The budget this returns is larger than the per-arch heuristic it replaced on every
    // arch, except where `kMaxShmemBytes` caps it. See there for the H200 regression.
    bytes = HistShmemBytes(device, Arch::kMinBlocks);
  });
  return bytes;
}

namespace {

__device__ GradientPairInt64 LoadGpair(GradientPairInt64 const* XGBOOST_RESTRICT gpairs) {
  static_assert(sizeof(int4) == sizeof(GradientPairInt64));
  auto g = *reinterpret_cast<int4 const*>(gpairs);
  return *reinterpret_cast<GradientPairInt64*>(&g);
}

// Build the histogram for the items [begin, end) of a single node, target, and feature group.
template <typename Policy, typename Accessor, typename RidxIterSpan>
__device__ void HistKernelSegment(Accessor const& matrix, FeatureGroup const& group,
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

  auto process_item = [&](auto idx) {
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

  // A single loop avoids carrying both a tile offset and an item offset through accumulation.
  for (bst_idx_t idx = begin + threadIdx.x; idx < end; idx += Policy::kBlockThreads) {
    process_item(idx);
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
// Ties resolve to the largest index, hence zero-width entries (empty nodes without
// padding) are skipped.
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
  // aligned. `volatile` keeps the position in thread-local memory, it is accessed only
  // between segments, allowing the flush metadata to be reconstructed without keeping it
  // in registers across the accumulation loop.
  volatile bst_idx_t pos = blockIdx.x * items_per_block;
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
      HistKernelSegment<Policy>(matrix, group, d_ridx_iters[seg.nidx_in_set],
                                d_gpair + n_samples * seg.target_idx, smem_hist, target_hist(seg),
                                seg.begin, seg.end);
      if constexpr (Policy::kSharedMem) {
        __syncthreads();
        // Recompute the segment instead of keeping it alive across the histogram loop,
        // which saves registers.
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

namespace {
// Stage the kernel metadata to the device. The host memory is pageable, the copy blocks
// until the staging is done, hence the caller can free the host memory upon return. The
// copy and the histogram kernel use the same stream.
template <typename T>
void CopyToDevice(Context const* ctx, std::vector<T> const& h_values,
                  dh::TemporaryArray<T>* p_out) {
  CHECK_EQ(p_out->size(), h_values.size());
  dh::safe_cuda(cudaMemcpyAsync(p_out->data().get(), h_values.data(), h_values.size() * sizeof(T),
                                cudaMemcpyHostToDevice, ctx->CUDACtx()->Stream()));
}
}  // namespace

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

  // Maps kernel instantiations to the number of resident blocks per MP. This is a mutable
  // state, as a result the histogram kernel is not thread safe.
  std::map<void*, std::int32_t> cfg;
  // The number of multi-processor for the selected GPU
  std::int32_t const n_mps;
  // Maximum size of the shared memory (optin)
  std::size_t const max_shared_bytes;
  // Use global memory for testing
  bool const force_global;

  // Obtain the (cached) number of resident blocks per MP for a kernel.
  template <typename Policy, typename Kernel>
  [[nodiscard]] std::int32_t BlocksPerMp(Policy, std::size_t shmem_bytes, Kernel kernel) {
    auto [it, inserted] = this->cfg.try_emplace(reinterpret_cast<void*>(kernel), 0);
    if (inserted) {
      if (shmem_bytes > 0) {
        // This function is the reason for all this trouble to cache the
        // configuration. It blocks the device.
        //
        // Also, it must precede the `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
        // otherwise the shmem bytes might be invalid.
        dh::safe_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           this->max_shared_bytes));
      }
      // Use this as a limiter, works for root node. Not too bad an option for child nodes.
      dh::safe_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &it->second, kernel, Policy::kBlockThreads, shmem_bytes));
      CHECK_GT(it->second, 0);
    }
    return it->second;
  }

  explicit HistKernel(Context const* ctx, bool force_global)
      : n_mps{curt::GetMpCnt(ctx->Ordinal())},
        max_shared_bytes{dh::MaxSharedMemoryOptin(ctx->Ordinal())},
        force_global{force_global} {}

  template <bool kDense, bool kCompressed, typename Accessor, typename RidxIterSpan>
  void DispatchHistShmem(Context const* ctx, Accessor const& matrix,
                         FeatureGroupsAccessor const& feature_groups,
                         linalg::MatrixView<GradientPairInt64 const> gpair,
                         std::vector<RidxIterSpan> const& h_ridx_iters,
                         std::vector<common::Span<GradientPairInt64>> const& h_hists) {
    CHECK(gpair.FContiguous());
    CHECK_EQ(h_ridx_iters.size(), h_hists.size());
    auto n_samples = gpair.Shape(0);
    auto n_targets = gpair.Shape(1);
    auto d_gpair = gpair.Values().data();

    // Cumulative sum of the number of rows in each node.
    std::vector<std::size_t> h_sizes_csum{0};
    h_sizes_csum.reserve(h_ridx_iters.size() + 1);
    for (auto const& ridx : h_ridx_iters) {
      h_sizes_csum.push_back(h_sizes_csum.back() + ridx.size());
    }
    if (h_sizes_csum.back() == 0) {
      return;
    }

    std::size_t shmem_bytes = feature_groups.ShmemSize();
    bool use_shared = !force_global && shmem_bytes <= this->max_shared_bytes;
    shmem_bytes = use_shared ? shmem_bytes : 0;

    // Stage the per-node metadata. The buffers are freed on the same stream as the kernel.
    dh::TemporaryArray<std::size_t> sizes_csum(h_sizes_csum.size());
    dh::TemporaryArray<RidxIterSpan> ridx_iters(h_ridx_iters.size());
    dh::TemporaryArray<common::Span<GradientPairInt64>> hists(h_hists.size());
    CopyToDevice(ctx, h_sizes_csum, &sizes_csum);
    CopyToDevice(ctx, h_ridx_iters, &ridx_iters);
    CopyToDevice(ctx, h_hists, &hists);

    auto launch = [&](auto policy) {
      using Policy = common::GetValueT<decltype(policy)>;
      auto kernel = HistogramKernel<Policy, Accessor, RidxIterSpan>;
      auto n_blocks_per_mp = this->BlocksPerMp(Policy{}, shmem_bytes, kernel);
      // Must match the kernel.
      bst_idx_t row_items =
          Policy::kCompressed ? matrix.row_stride : matrix.row_stride * feature_groups.NumGroups();
      bst_idx_t n_segments = (h_sizes_csum.size() - 1) * feature_groups.NumGroups();
      bst_idx_t n_items =
          n_targets * (h_sizes_csum.back() * row_items + n_segments * Policy::kSegmentCost);
      auto [items_per_block, n_blocks] = SliceItems<Policy>(n_items, n_blocks_per_mp * n_mps);
      dh::LaunchKernel(n_blocks, Policy::kBlockThreads, shmem_bytes, ctx->CUDACtx()->Stream())(
          kernel, matrix, feature_groups, ridx_iters.data().get(), dh::ToSpan(sizes_csum),
          hists.data().get(), d_gpair, n_samples, n_targets, items_per_block, n_items);
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
    dispatch_arch([&](auto arch) {
      using Arch = common::GetValueT<decltype(arch)>;
      if (use_shared) {
        launch(HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, true>{});
      } else {
        launch(HistPolicy<Arch, kItemsPerThread, kDense, kCompressed, false>{});
      }
    });
  }

  template <typename Accessor, typename... Args>
  void Dispatch(Context const* ctx, Accessor const& matrix, Args&&... args) {
    if (matrix.IsDense()) {
      DispatchHistShmem<true, true>(ctx, matrix, std::forward<Args>(args)...);
    } else if (matrix.IsDenseCompressed()) {
      DispatchHistShmem<false, true>(ctx, matrix, std::forward<Args>(args)...);
    } else {
      DispatchHistShmem<false, false>(ctx, matrix, std::forward<Args>(args)...);
    }
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
                      std::vector<common::Span<cuda_impl::RowIndexT const>> const& ridxs,
                      std::vector<common::Span<GradientPairInt64>> const& hists) {
    if (ridxs.size() == 1 && ridxs.front().size() == matrix.n_rows) {
      // Special optimization for the root node, the row index is the identity mapping.
      using RidxIter = dh::counting_iterator<cuda_impl::RowIndexT>;
      CHECK_LT(matrix.base_rowid, std::numeric_limits<cuda_impl::RowIndexT>::max());
      std::vector<common::IterSpan<RidxIter>> ridx_iters{common::IterSpan{
          dh::make_counting_iterator(static_cast<cuda_impl::RowIndexT>(matrix.base_rowid)),
          matrix.n_rows}};
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridx_iters, hists);
    } else {
      this->kernel_->Dispatch(ctx, matrix, feature_groups, gpair, ridxs, hists);
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
  this->BuildHistogram(ctx, matrix, feature_groups,
                       linalg::MakeTensorView(ctx, linalg::kF, gpair, gpair.size(), 1), {ridx},
                       {histogram});
}

void DeviceHistogramBuilder::BuildHistogram(
    Context const* ctx, EllpackAccessor const& matrix, FeatureGroupsAccessor const& feature_groups,
    linalg::MatrixView<GradientPairInt64 const> gpair,
    std::vector<common::Span<cuda_impl::RowIndexT const>> const& ridxs,
    std::vector<common::Span<GradientPairInt64>> const& hists) {
  this->monitor_.Start(__func__);
  std::visit(
      [&](auto&& matrix) {
        this->p_impl_->BuildHistogram(ctx, matrix, feature_groups, gpair, ridxs, hists);
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
