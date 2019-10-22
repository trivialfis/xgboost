/*!
 * Copyright 2015 by Contributors
 * \file tree_updater.cc
 * \brief Registry of tree updaters.
 */
#include <dmlc/registry.h>

#include "xgboost/tree_updater.h"
#include "xgboost/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::TreeUpdaterReg);
}  // namespace dmlc

namespace xgboost {

TreeUpdater* TreeUpdater::Create(const std::string& name, LeaveIndexCache* cache,
                                 GenericParameter const* tparam) {
  auto *e = ::dmlc::Registry< ::xgboost::TreeUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown tree updater " << name;
  }
  auto p_updater = (e->body)();
  p_updater->tparam_ = tparam;
  p_updater->SetCache(cache);
  return p_updater;
}

common::Span<size_t const> TreeUpdater::LeaveIndexCache::operator[](unsigned node_id) const {
  return (*const_cast<LeaveIndexCache*>(this))[node_id];
}

common::Span<size_t> TreeUpdater::LeaveIndexCache::operator[](unsigned node_id) {
  Segment seg { ridx_segments.at(node_id) };
  // Avoids calling __host__ inside __host__ __device__
  auto* ptr = row_index.HostVector().data();
  auto size = row_index.HostVector().size();
  if (seg.begin == seg.end) {
    return common::Span<size_t>{};
  } else {
    return common::Span<size_t>{ptr, size}.subspan(
        seg.begin, seg.end - seg.begin);
  }
}

void TreeUpdater::LeaveIndexCache::AddSplit(unsigned node_id, size_t iLeft, unsigned left_node_id,
                                            unsigned right_node_id) {
  Segment const& e = ridx_segments[node_id];

  CHECK(e.end != 0);

  size_t begin = e.begin;
  size_t split_pt = begin + iLeft;

  if (left_node_id >= ridx_segments.size()) {
    ridx_segments.resize((left_node_id + 1) * 2, Segment(0, 0));
  }
  if (right_node_id >= ridx_segments.size()) {
    ridx_segments.resize((right_node_id + 1) * 2, Segment(0, 0));
  }

  // ridx_segments.
  ridx_segments[left_node_id] = Segment(begin, split_pt);
  ridx_segments[right_node_id] = Segment(split_pt, e.end);
  ridx_segments[node_id] = Segment(begin, e.end);
}

}  // namespace xgboost

namespace xgboost {
namespace tree {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_colmaker);
DMLC_REGISTRY_LINK_TAG(updater_skmaker);
DMLC_REGISTRY_LINK_TAG(updater_refresh);
DMLC_REGISTRY_LINK_TAG(updater_prune);
DMLC_REGISTRY_LINK_TAG(updater_quantile_hist);
DMLC_REGISTRY_LINK_TAG(updater_histmaker);
DMLC_REGISTRY_LINK_TAG(updater_sync);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu_hist);
#endif  // XGBOOST_USE_CUDA
}  // namespace tree
}  // namespace xgboost
