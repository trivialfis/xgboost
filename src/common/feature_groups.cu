/**
 * Copyright 2020-2026, XGBoost Contributors
 */

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#include <vector>     // for vector

#include "device_helpers.cuh"
#include "feature_groups.cuh"
#include "hist_util.h"  // for HistogramCuts

namespace xgboost::common {
FeatureGroups::FeatureGroups(common::HistogramCuts const& cuts, bool is_dense, size_t shm_size)
    : max_group_bins{0} {
  // Only use a single feature group for sparse matrices.
  bool single_group = !is_dense;
  if (single_group) {
    InitSingle(cuts);
    return;
  }

  auto& feature_segments_h = feature_segments.HostVector();
  auto& bin_segments_h = bin_segments.HostVector();
  feature_segments_h.push_back(0);
  bin_segments_h.push_back(0);

  std::vector<std::uint32_t> const& cut_ptrs = cuts.Ptrs();
  // Maximum number of bins that can be placed into shared memory (single target).
  std::size_t max_shmem_bins = shm_size / sizeof(GradientPairInt64);

  for (size_t i = 2; i < cut_ptrs.size(); ++i) {
    int last_start = bin_segments_h.back();
    // Push a new group whenever the size of required bin storage is greater than the
    // shared memory size.
    if (cut_ptrs[i] - last_start > max_shmem_bins) {
      feature_segments_h.push_back(i - 1);
      bin_segments_h.push_back(cut_ptrs[i - 1]);
      max_group_bins = std::max(max_group_bins, bin_segments_h.back() - last_start);
    }
  }
  feature_segments_h.push_back(cut_ptrs.size() - 1);
  bin_segments_h.push_back(cut_ptrs.back());
  max_group_bins =
      std::max(max_group_bins, bin_segments_h.back() - bin_segments_h[bin_segments_h.size() - 2]);
  this->InitFeatureMap();
}

void FeatureGroups::InitSingle(common::HistogramCuts const& cuts) {
  auto& feature_segments_h = feature_segments.HostVector();
  feature_segments_h.push_back(0);
  feature_segments_h.push_back(cuts.Ptrs().size() - 1);

  auto& bin_segments_h = bin_segments.HostVector();
  bin_segments_h.push_back(0);
  bin_segments_h.push_back(cuts.TotalBins());

  max_group_bins = cuts.TotalBins();
}

FeatureGroups::FeatureGroups(HistogramCuts const& cuts, std::vector<bst_feature_t> segments)
    : max_group_bins{0} {
  CHECK_GE(segments.size(), 2);
  CHECK_EQ(segments.front(), 0);
  CHECK_EQ(segments.back(), cuts.NumFeatures());
  auto const& ptrs = cuts.Ptrs();
  auto& bins = this->bin_segments.HostVector();
  for (std::size_t i = 0; i < segments.size(); ++i) {
    if (i != 0) {
      CHECK(segments[i] > segments[i - 1] || cuts.NumFeatures() == 0);
    }
    bins.push_back(ptrs[segments[i]]);
    if (i != 0) {
      this->max_group_bins = std::max(this->max_group_bins, bins[i] - bins[i - 1]);
    }
  }
  this->feature_segments.HostVector() = std::move(segments);
  this->InitFeatureMap();
}

void FeatureGroups::InitFeatureMap() {
  auto const& segments = this->feature_segments.ConstHostVector();
  if (segments.size() <= 2) {
    return;
  }
  auto& index = this->feature_group_index.HostVector();
  index.resize(segments.back());
  for (std::size_t g = 0; g + 1 < segments.size(); ++g) {
    std::fill(index.begin() + segments[g], index.begin() + segments[g + 1],
              FeatureGroupIndex{segments[g], segments[g + 1] - segments[g]});
  }
}

namespace {
// The largest budget that preserves the requested number of co-resident blocks.
std::size_t HistShmemBytes(std::int32_t device, std::int32_t min_blocks) {
  constexpr std::size_t kMaxShmemBytes = 96 * 1024;
  constexpr std::int32_t kShmemAllocGranularity = 128;
  auto optin = dh::MaxSharedMemoryOptin(device);
  std::int32_t smem_per_sm = 0, reserved = 0;
  dh::safe_cuda(
      cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  dh::safe_cuda(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device));
  auto per_block =
      (smem_per_sm / min_blocks / kShmemAllocGranularity) * kShmemAllocGranularity - reserved;
  CHECK_GT(per_block, 0);
  return std::min({static_cast<std::size_t>(per_block), optin, kMaxShmemBytes});
}
}  // namespace

std::size_t DftStHistShmemBytes(std::int32_t device) {
  std::int32_t max_threads_per_sm = 0;
  dh::safe_cuda(
      cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
  return HistShmemBytes(device, std::max(1, max_threads_per_sm / 1024));
}

std::size_t DftMtHistShmemBytes(std::int32_t device) {
  std::int32_t version = 0;
  dh::safe_cuda(cub::SmVersion(version, device));
  // Matches HistSm75 and the HistSm80-and-newer policies in gpu_hist/histogram.cu.
  return HistShmemBytes(device, version >= 800 ? 2 : 1);
}
}  // namespace xgboost::common
