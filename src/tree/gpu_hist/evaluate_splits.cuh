/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_GPU_HIST_EVALUATE_SPLITS_CUH_
#define XGBOOST_TREE_GPU_HIST_EVALUATE_SPLITS_CUH_

#include <memory>  // for unique_ptr, make_unique
#include <vector>  // for vector

#include "../../common/categorical.h"            // for CatWordT, IsCat
#include "../../common/cuda_pinned_allocator.h"  // for PinnedAllocator
#include "../../common/cuda_stream.h"            // for Stream
#include "../../common/device_helpers.cuh"       // for WarpThreads
#include "../../common/device_vector.cuh"        // for DeviceUVector
#include "../split_evaluator.h"                  // for TreeEvaluator
#include "expand_entry.cuh"                      // for GPUExpandEntry, MultiExpandEntry
#include "quantiser.cuh"                         // for GradientQuantiser

namespace xgboost::common {
class HistogramCuts;
}

namespace xgboost::tree {
// Butterfly all-reduce within a warp. Every lane receives the full sum.
XGBOOST_DEV_INLINE
GradientPairInt64 WarpSum(GradientPairInt64 v) {
  for (std::int32_t offset = dh::WarpThreads() / 2; offset > 0; offset >>= 1) {
    auto g = __shfl_xor_sync(dh::WarpFullMask(), v.GetQuantisedGrad(), offset);
    auto h = __shfl_xor_sync(dh::WarpFullMask(), v.GetQuantisedHess(), offset);
    v += GradientPairInt64{g, h};
  }
  return v;
}

// Inputs specific to each node
struct EvaluateSplitInputs {
  bst_node_t nidx;
  bst_node_t depth;
  GradientPairInt64 parent_sum;
  common::Span<const bst_feature_t> feature_set;
  common::Span<const GradientPairInt64> gradient_histogram;
};

// Inputs necessary for all nodes
struct EvaluateSplitSharedInputs {
  EvalParam param;
  GradientQuantiser rounding;
  common::Span<FeatureType const> feature_types;
  common::Span<const uint32_t> feature_segments;
  common::Span<const float> feature_values;
  bool is_dense;
  [[nodiscard]] XGBOOST_DEVICE auto Features() const { return feature_segments.size() - 1; }
  [[nodiscard]] __device__ std::uint32_t FeatureBins(bst_feature_t fidx) const {
    return feature_segments[fidx + 1] - feature_segments[fidx];
  }
};

// Input for evaluation kernel for each tree node.
struct MultiEvaluateSplitInputs {
  bst_node_t nidx;
  bst_node_t depth;
  common::Span<GradientPairInt64 const> parent_sum;
  common::Span<bst_feature_t const> feature_set;
  common::Span<GradientPairInt64 const> histogram;
};

// Input for evaluation kernel that can be shared by multiple tree nodes.
struct MultiEvaluateSplitSharedInputs {
  // len == n_split_targets
  common::Span<GradientQuantiser const> roundings;
  // cut pointers
  common::Span<std::uint32_t const> feature_segments;
  // cut values
  float const *feature_values;
  common::Span<FeatureType const> feature_types;
  // Number of bit-field storage elements required for one categorical split.
  std::size_t cat_storage_size{0};
  // Number of histogram bins for one target, across all features.
  bst_bin_t n_total_bins_per_tar;
  bst_feature_t max_active_feature;
  EvalParam param;

  // Used for testing
  enum OnePass {
    kNone,      // normal
    kForward,   // only perform the forward pass
    kBackward,  // only perform the backward pass
  } one_pass{kNone};

  // A globally dense matrix needs only the forward numerical scan.
  bool is_dense{false};

  [[nodiscard]] XGBOOST_DEVICE bool NeedBackward() const {
    return one_pass != kForward && (!is_dense || one_pass == kBackward);
  }

  [[nodiscard]] XGBOOST_DEVICE bst_target_t Targets() const { return roundings.size(); }
  [[nodiscard]] XGBOOST_DEVICE bst_feature_t Features() const {
    return this->feature_segments.size() - 1;
  }
  [[nodiscard]] XGBOOST_DEVICE bool IsCategorical(bst_feature_t fidx) const {
    return common::IsCat(this->feature_types, fidx);
  }
};

namespace cuda_impl {
/** @brief Shared scalar and vector GPU histogram split evaluation. */
class GPUHistEvaluator {
 public:
  template <typename GradT>
  static XGBOOST_DEVICE common::Span<GradT> GetNodeSumImpl(common::Span<GradT> node_sums,
                                                           bst_node_t nidx,
                                                           bst_target_t n_split_targets) {
    auto offset = nidx * n_split_targets;
    return node_sums.subspan(offset, n_split_targets);
  }

  /** @brief Buffer to access node weights indexed by node id. */
  struct NodeWeightBuffer {
    // * 3 because of base, left, right weights per node.
    constexpr static bst_node_t kWeightsPerNode = 3;

    common::Span<float> weights;
    bst_target_t n_split_targets;

    // Get the base weight buffer for a node
    [[nodiscard]] XGBOOST_DEVICE common::Span<float> Base(bst_node_t nidx) const {
      return weights.subspan(nidx * n_split_targets * kWeightsPerNode, n_split_targets);
    }
    // Get the left child weight buffer for a node
    [[nodiscard]] XGBOOST_DEVICE common::Span<float> Left(bst_node_t nidx) const {
      return weights.subspan(nidx * n_split_targets * kWeightsPerNode + n_split_targets,
                             n_split_targets);
    }
    // Get the right child weight buffer for a node
    [[nodiscard]] XGBOOST_DEVICE common::Span<float> Right(bst_node_t nidx) const {
      return weights.subspan(nidx * n_split_targets * kWeightsPerNode + n_split_targets * 2,
                             n_split_targets);
    }
  };

  struct NodeSumBuffer {
    dh::DeviceUVector<GradientPairInt64> node_sums;

    /**
     * @brief Allocate storage for node sums up to the given node ID.
     */
    void Alloc(bst_node_t nidx, bst_target_t n_split_targets) {
      auto end = (nidx + 1) * n_split_targets;
      if (this->node_sums.size() < end) {
        this->node_sums.resize(end);
      }
    }
    [[nodiscard]] common::Span<GradientPairInt64> GetNode(bst_node_t nidx,
                                                          bst_target_t n_split_targets) {
      return GetNodeSumImpl(dh::ToSpan(this->node_sums), nidx, n_split_targets);
    }
    [[nodiscard]] common::Span<GradientPairInt64 const> GetNode(
        bst_node_t nidx, bst_target_t n_split_targets) const {
      return GetNodeSumImpl(dh::ToSpan(this->node_sums), nidx, n_split_targets);
    }
    auto View() { return dh::ToSpan(this->node_sums); }
    auto View() const { return dh::ToSpan(this->node_sums); }
  };

 private:
  std::unique_ptr<TreeEvaluator> tree_evaluator_{nullptr};
  // Scalar tree expansion needs host category bits as well as device bits for partitioning.
  using CatAllocator = common::cuda_impl::PinnedAllocator<CatWordT>;
  std::vector<CatWordT, CatAllocator> h_split_cats_;
  curt::Stream copy_stream_;
  void CopyToHost(Context const *ctx, std::vector<bst_node_t> const &nidx);

  // Persistent buffer for node weights, indexed by node id.
  dh::DeviceUVector<float> node_weights_;
  // Buffer for histogram scans.
  dh::DeviceUVector<GradientPairInt64> scan_buffer_;
  // Buffer for node gradient sums. Nodes stored in this buffer are valid nodes (exist in
  // the output tree) instead of candidates.
  NodeSumBuffer node_sums_;
  // Buffer for split sums (child_sum at split point), indexed by node id. This temporary
  // buffer is needed because we don't have the child node index during evaluation, which
  // is only available after applying split to the tree.
  NodeSumBuffer split_sums_;
  // Category bit fields for evaluated nodes, indexed by node id. They must remain valid
  // while candidates are waiting in the loss-guide queue.
  dh::DeviceUVector<CatWordT> split_cats_;
  std::size_t node_cat_storage_size_{0};
  // Whether any categorical feature requires partition-based evaluation.
  bool need_sort_histogram_{false};

  void AllocNodeCats(bst_node_t nidx, std::size_t storage_size) {
    if (this->node_cat_storage_size_ == 0) {
      this->node_cat_storage_size_ = storage_size;
    }
    CHECK_EQ(this->node_cat_storage_size_, storage_size);
    auto required = (nidx + 1) * storage_size;
    if (this->split_cats_.size() < required) {
      this->split_cats_.resize(required);
    }
  }

 public:
  GPUHistEvaluator() = default;
  GPUHistEvaluator(TrainParam const &param, bst_feature_t n_features, DeviceOrd device)
      : tree_evaluator_{std::make_unique<TreeEvaluator>(param, n_features, device, 1u)} {}
  ~GPUHistEvaluator() {
    // Finish category transfers before releasing either host or device storage.
    copy_stream_.View().Sync();
  }

  // Adapters for scalar expansion entries. All split enumeration uses the common pipeline.
  void Reset(Context const *ctx, common::HistogramCuts const &cuts,
             common::Span<FeatureType const> feature_types, bst_feature_t n_features,
             TrainParam const &param);
  void EvaluateSplits(Context const *ctx, std::vector<bst_node_t> const &nidx,
                      bst_feature_t max_active_features,
                      common::Span<EvaluateSplitInputs const> inputs,
                      EvaluateSplitSharedInputs shared, common::Span<GPUExpandEntry> out);
  GPUExpandEntry EvaluateSingleSplit(Context const *ctx, EvaluateSplitInputs input,
                                     EvaluateSplitSharedInputs shared);
  void ApplyTreeSplit(Context const *ctx, GPUExpandEntry const &candidate, RegTree *tree);

  [[nodiscard]] common::Span<CatWordT const> GetHostNodeCats(bst_node_t nidx) const {
    copy_stream_.View().Sync();
    return common::Span<CatWordT const>{h_split_cats_}.subspan(nidx * node_cat_storage_size_,
                                                               node_cat_storage_size_);
  }
  [[nodiscard]] common::KCatBitField GetDeviceNodeCats(bst_node_t nidx) const {
    copy_stream_.View().Sync();
    return common::KCatBitField{this->GetNodeCats(nidx)};
  }

  void Reset(Context const *ctx, common::Span<std::uint32_t const> feature_segments,
             common::Span<FeatureType const> feature_types, TrainParam const &param,
             bst_target_t n_split_targets);

  [[nodiscard]] auto GetEvaluator() const { return tree_evaluator_->GetEvaluator<EvalParam>(); }

  /**
   * @brief Run evaluation for the root node.
   */
  [[nodiscard]] MultiExpandEntry EvaluateSingleSplit(
      Context const *ctx, MultiEvaluateSplitInputs const &input,
      MultiEvaluateSplitSharedInputs const &shared_inputs);
  /**
   * @brief Run evaluation for multiple nodes.
   *
   * @param max_nidx Maximum node ID among the nodes being evaluated. Used to allocate
   *                 weight storage on demand.
   */
  void EvaluateSplits(Context const *ctx, common::Span<MultiEvaluateSplitInputs const> d_inputs,
                      MultiEvaluateSplitSharedInputs const &shared_inputs, bst_node_t max_nidx,
                      common::Span<MultiExpandEntry> out_splits);

  /**
   * @brief Allocate storage for node sums up to the given node ID.
   */
  void AllocNodeSum(bst_node_t nidx, bst_target_t n_split_targets) {
    this->node_sums_.Alloc(nidx, n_split_targets);
  }
  [[nodiscard]] common::Span<GradientPairInt64> GetNodeSum(bst_node_t nidx,
                                                           bst_target_t n_split_targets) {
    return this->node_sums_.GetNode(nidx, n_split_targets);
  }

  /**
   * @brief Allocate storage for weights up to the given node ID.
   */
  void AllocNodeWeight(bst_node_t nidx, bst_target_t n_split_targets) {
    auto required = (nidx + 1) * n_split_targets * NodeWeightBuffer::kWeightsPerNode;
    if (this->node_weights_.size() < required) {
      this->node_weights_.resize(required);
    }
  }
  [[nodiscard]] NodeWeightBuffer GetNodeWeights(bst_target_t n_split_targets) {
    return NodeWeightBuffer{dh::ToSpan(this->node_weights_), n_split_targets};
  }
  [[nodiscard]] common::Span<CatWordT const> GetNodeCats(bst_node_t nidx) const {
    if (this->node_cat_storage_size_ == 0) {
      return {};
    }
    return dh::ToSpan(this->split_cats_)
        .subspan(nidx * this->node_cat_storage_size_, this->node_cat_storage_size_);
  }

  // Update the tree evaluator state and track child gradient sums.
  void ApplyTreeSplit(Context const *ctx, RegTree const *p_tree,
                      common::Span<MultiExpandEntry const> h_candidates,
                      common::Span<MultiExpandEntry const> d_candidates,
                      bst_target_t n_split_targets);
};
using MultiHistEvaluator = GPUHistEvaluator;
}  // namespace cuda_impl
using GPUHistEvaluator = cuda_impl::GPUHistEvaluator;
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_GPU_HIST_EVALUATE_SPLITS_CUH_
