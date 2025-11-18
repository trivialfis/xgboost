/**
 * Copyright 2025, XGBoost contributors
 */
#include <vector>

#include "../common/cuda_context.cuh"
#include "../common/deterministic.cuh"
#include "../common/linalg_op.cuh"
#include "../data/batch_utils.h"
#include "../data/ellpack_page.cuh"
#include "../tree/gpu_hist/quantiser.cuh"
#include "../tree/gpu_hist/row_partitioner.cuh"
#include "../tree/updater_gpu_hist.cuh"
#include "xgboost/data.h"
#include "xgboost/gradient.h"
#include "xgboost/tree_model.h"

namespace xgboost::cv {
// todos:
// - intercepts
// - build histogram
// - evaluation
// - partition

using xgboost::cuda_impl::StaticBatch;
namespace {
// fixme: duplicated code
struct Pair {
  GradientPair first;
  GradientPair second;
};
__host__ XGBOOST_DEV_INLINE Pair operator+(Pair const& lhs, Pair const& rhs) {
  return {lhs.first + rhs.first, lhs.second + rhs.second};
}
struct Clip {
  static XGBOOST_DEV_INLINE float Pclip(float v) { return v > 0 ? v : 0; }
  static XGBOOST_DEV_INLINE float Nclip(float v) { return v < 0 ? abs(v) : 0; }

  XGBOOST_DEV_INLINE Pair operator()(GradientPair x) const {
    auto pg = Pclip(x.GetGrad());
    auto ph = Pclip(x.GetHess());

    auto ng = Nclip(x.GetGrad());
    auto nh = Nclip(x.GetHess());

    return {GradientPair{pg, ph}, GradientPair{ng, nh}};
  }
};

tree::GradientQuantiser* CreateQuantizer(Pair p, bst_idx_t total_rows) {
  using GradientSumT = GradientPairPrecise;
  using T = typename GradientSumT::ValueT;
  GradientPair positive_sum{p.first}, negative_sum{p.second};

  auto histogram_rounding =
      GradientSumT{common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetGrad(), negative_sum.GetGrad()), total_rows),
                   common::CreateRoundingFactor<T>(
                       std::max(positive_sum.GetHess(), negative_sum.GetHess()), total_rows)};

  using IntT = typename GradientPairInt64::ValueT;

  /**
   * Factor for converting gradients from fixed-point to floating-point.
   */
  auto to_floating_point_ =
      histogram_rounding /
      static_cast<T>(static_cast<IntT>(1)
                     << (sizeof(typename GradientSumT::ValueT) * 8 - 2));  // keep 1 for sign bit
  /**
   * Factor for converting gradients from floating-point to fixed-point. For
   * f64:
   *
   *   Precision = 64 - 1 - log2(rounding)
   *
   * rounding is calcuated as exp(m), see the rounding factor calcuation for
   * details.
   */
  auto to_fixed_point_ = GradientSumT(static_cast<T>(1) / to_floating_point_.GetGrad(),
                                      static_cast<T>(1) / to_floating_point_.GetHess());
  return new tree::GradientQuantiser{to_fixed_point_, to_floating_point_};
}

// fixme: copy duplication

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

template <typename IterT>
XGBOOST_DEV_INLINE bst_idx_t IterIdx(EllpackAccessorImpl<IterT> const& matrix, bst_idx_t base_rowid,
                                     tree::RowPartitioner::RowIndexT ridx, bst_feature_t fidx) {
  // ridx_local = ridx - base_rowid  <== Row index local to each batch
  // entry_idx = ridx_local * row_stride <== Starting entry index for this row in the matrix
  // entry_idx += start_feature  <== Inside a row, first column inside this feature group
  // idx % feature_stride <== The feaature index local to the current feature group
  // entry_idx += idx % feature_stride <== Final index.
  return (ridx - base_rowid) * matrix.row_stride + fidx;
}

template <typename Accessor>
__global__ void MultiHistKernel(Accessor const matrix, bst_idx_t base_rowid,
                                common::Span<tree::RowPartitioner::RowIndexT const> d_ridx,
                                GradientPairInt64* d_node_hist,
                                linalg::MatrixView<const GradientPair> d_gpair,
                                common::Span<tree::GradientQuantiser const> roundings) {
  std::int32_t feature_stride = matrix.row_stride;
  bst_idx_t n_elements = feature_stride * d_ridx.size();
  using Idx = tree::RowPartitioner::RowIndexT;
  for (auto idx : dh::GridStrideRange(static_cast<std::size_t>(0), n_elements)) {
    Idx ridx = d_ridx[idx / feature_stride];
    auto fidx = idx % feature_stride;
    bst_bin_t compressed_bin = matrix.gidx_iter[IterIdx(matrix, base_rowid, ridx, fidx)];
    if (compressed_bin != matrix.NullValue()) {
      // bool kCompressed
      // fixme
      if (true) {
        compressed_bin += matrix.feature_segments[fidx];
      }
      bst_target_t n_targets = roundings.size();
      compressed_bin *= n_targets;
      // TODO(jiamingy): Assign a thread for each target.
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto adjusted = roundings[t].ToFixedPoint(d_gpair(ridx, t));
        AtomicAddGpairGlobal(d_node_hist + compressed_bin + t, adjusted);
      }
    }
  }
}

struct BatchPtr {
  std::vector<std::vector<bst_idx_t>> batch_ptr;
  BatchPtr(std::int32_t n_batches, std::int32_t n_folds) : batch_ptr(n_batches + 1) {
    CHECK_GE(n_batches, 1);
    batch_ptr[0] = std::vector<bst_idx_t>(n_folds, 0);
  }
  std::vector<bst_idx_t>& Batch(std::int32_t fold_idx) { return batch_ptr.at(fold_idx); }
  void InclusiveSum() {
    for (std::size_t fold_idx = 0; fold_idx < batch_ptr.front().size(); ++fold_idx) {
      for (std::size_t batch_idx = 1; batch_idx < batch_ptr.size(); ++batch_idx) {
        auto size = batch_ptr[batch_idx - 1].at(fold_idx);
        batch_ptr[batch_idx].at(fold_idx) += size;
      }
    }
  }
};
}  // namespace

// Maybe we can modify the multi-target builder to handle many trees
void BuildTrees(Context const* ctx, DMatrix* p_fmat,
                std::vector<std::vector<std::unique_ptr<GradientContainer>>> const& gpairs,
                std::vector<std::vector<std::vector<bst_idx_t>>> const& tr_idx,
                std::vector<RegTree*> trees) {
  auto n_folds = trees.size();
  auto n_targets = trees.front()->NumTargets();

  // Init data
  // each fold needs a different quantizer
  std::vector<std::unique_ptr<tree::GradientQuantiser>> split_quantizer;
  std::vector<Pair> running_sum(n_folds * n_targets);
  std::vector<bst_idx_t> running_sum_rows(n_folds);
  for (std::int32_t batch_idx = 0; batch_idx < p_fmat->NumBatches(); ++batch_idx) {
    auto const& batch_gpairs = gpairs.at(batch_idx);
    for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto fold_gpair = batch_gpairs.at(fold_idx)->gpair.View(ctx->Device());
      auto beg = thrust::make_transform_iterator(linalg::tcbegin(fold_gpair), Clip());
      for (bst_target_t t = 0; t < n_targets; ++t) {
        running_sum[fold_idx * n_targets + t] =
            dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + fold_gpair.Size(),
                       running_sum[fold_idx * n_targets + t], thrust::plus<Pair>{});
      }
      running_sum_rows[fold_idx] += fold_gpair.Shape<0>();
    }
  }
  for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    for (bst_target_t t = 0; t < n_targets; ++t) {
      split_quantizer.emplace_back(
          CreateQuantizer(running_sum[fold_idx * n_targets + t], running_sum_rows[fold_idx]));
    }
  }

  // Accumulate the root sum from all batches
  // Init root
  std::int32_t batch_idx = 0;
  dh::device_vector<GradientPairInt64> root_sums(n_folds * n_targets);

  // fixme: find a better ds.
  BatchPtr batch_ptr(p_fmat->NumBatches(), n_folds);

  std::shared_ptr<common::HistogramCuts const> cuts;

  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
    cuts = page.Impl()->CutsShared();

    auto const& batch_gpairs = gpairs.at(batch_idx);
    auto const& batch_tr_idx = tr_idx.at(batch_idx);

    auto& local_ptr = batch_ptr.Batch(batch_idx + 1);

    for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto d_gpair = batch_gpairs.at(fold_idx)->gpair.View(ctx->Device());
      // We can use d_gpair without permutation indexing as it's calculated from the fold.
      auto fold_root_sum = dh::ToSpan(root_sums).subspan(fold_idx * n_targets, n_targets);
      // fixme: perf
      dh::device_vector<tree::GradientQuantiser> d_q;
      for (bst_target_t t = 0; t < n_targets; ++t) {
        d_q.push_back(*split_quantizer.at(fold_idx * n_targets + t));
      }
      tree::cuda_impl::CalcRootSum(ctx, d_gpair, dh::ToSpan(d_q), fold_root_sum);
      auto const& fold_tr_idx = batch_tr_idx.at(fold_idx);
      local_ptr.push_back(fold_tr_idx.size());
    }

    ++batch_idx;
  }
  batch_ptr.InclusiveSum();

  // Initialize partitioners
  std::vector<std::unique_ptr<tree::RowPartitioner>> partitioners;
  for (std::int32_t batch_idx = 0; batch_idx < p_fmat->NumBatches(); ++batch_idx) {
    // auto const& local_ptr = batch_ptr.Fold(batch_idx);
    for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      partitioners.emplace_back(std::make_unique<tree::RowPartitioner>());
      auto base_rowid = batch_ptr.Batch(batch_idx).at(fold_idx);
      auto fold_size = batch_ptr.Batch(batch_idx + 1).at(fold_idx) - base_rowid;
      partitioners.back()->Reset(ctx, fold_size, base_rowid);
    }
  }
  CHECK_EQ(partitioners.size(), n_folds * p_fmat->NumBatches());

  // Build root histogram.
  std::vector<tree::DeviceHistogramBuilder> histogram_builders(n_folds);
  tree::HistMakerTrainParam hist_param;
  auto feature_groups = std::make_unique<tree::FeatureGroups>(*cuts, true, 0ul);
  hist_param.UpdateAllowUnknown(Args{{}});
  for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    histogram_builders.at(fold_idx).Reset(ctx, hist_param.MaxCachedHistNodes(ctx->Device()),
                                          feature_groups->DeviceAccessor(ctx->Device()),
                                          cuts->TotalBins(), false);
    histogram_builders.at(fold_idx).AllocateHistograms(ctx, {RegTree::kRoot});
  }

  batch_idx = 0;
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
    auto const& batch_gpairs = gpairs.at(batch_idx);
    auto const& batch_tr_idx = tr_idx.at(batch_idx);  // fixme: find batch local idx
    auto batch = page.Impl();
    for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto d_ridx = partitioners.at(batch_idx * n_folds + fold_idx)->GetRows(RegTree::kRoot);
      auto d_node_hist = histogram_builders.at(fold_idx).GetNodeHistogram(RegTree::kRoot);
      auto d_gpair = batch_gpairs.at(fold_idx)->gpair.View(ctx->Device());
      auto roundings = *split_quantizer.at(fold_idx);
      auto base_rowid = batch_ptr.Batch(batch_idx).at(fold_idx);
      dh::device_vector<tree::GradientQuantiser> d_roundings{roundings};  // fixme
      batch->Visit(ctx, {}, [&](auto&& d_acc) {
        using Accessor = common::GetValueT<decltype(d_acc)>;
        constexpr std::uint32_t kBlockThreads = 512;
        // fixme
        std::uint32_t grid_size = std::min(d_ridx.size() * d_acc.row_stride / kBlockThreads, 4ul);
        MultiHistKernel<<<grid_size, kBlockThreads>>>(d_acc, base_rowid, d_ridx, d_node_hist.data(),
                                                      d_gpair, dh::ToSpan(d_roundings));
      });
    }

    ++batch_idx;
  }

  // Evaluate root split
  std::vector<std::unique_ptr<tree::cuda_impl::MultiHistEvaluator>> evaluators;
  tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{}});
  tree::GPUTrainingParam gpu_param{param};

  std::vector<tree::cuda_impl::MultiExpandEntry> root_entries;

  for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    evaluators.emplace_back(std::make_unique<tree::cuda_impl::MultiHistEvaluator>());

    auto node_hist = histogram_builders.at(fold_idx).GetNodeHistogram(RegTree::kRoot);
    auto p_tree = trees[fold_idx];

    auto fold_root_sum = dh::ToSpan(root_sums).subspan(fold_idx * n_targets, n_targets);
    evaluators.back()->AllocNodeSum(RegTree::kRoot, n_targets);
    auto d_root_sum = evaluators.back()->GetNodeSum(RegTree::kRoot, n_targets);
    dh::safe_cuda(cudaMemcpyAsync(d_root_sum.data(), fold_root_sum.data(), d_root_sum.size_bytes(),
                                  cudaMemcpyDefault, ctx->CUDACtx()->Stream()));

    tree::MultiEvaluateSplitInputs input{RegTree::kRoot, p_tree->GetDepth(RegTree::kRoot),
                                         fold_root_sum, node_hist};
    auto roundings = *split_quantizer.at(fold_idx);
    dh::device_vector<tree::GradientQuantiser> d_roundings{roundings};  // fixme

    tree::MultiEvaluateSplitSharedInputs shared_inputs{dh::ToSpan(d_roundings),
                                                       cuts->cut_ptrs_.ConstDeviceSpan(),
                                                       cuts->cut_values_.ConstDeviceSpan(),
                                                       cuts->min_vals_.ConstDeviceSpan(),
                                                       param.max_bin,
                                                       gpu_param};
    auto entry = evaluators.at(fold_idx)->EvaluateSingleSplit(ctx, input, shared_inputs);
    root_entries.push_back(entry);

    // TODO(jiamingy): Support learning rate.
    // TODO(jiamingy): We need to modify the tree structure to account for internal reduced weight
    // size.
    std::vector<float> h_base_weight(entry.base_weight.size());
    dh::CopyDeviceSpanToVector(&h_base_weight, entry.base_weight);
    p_tree->SetRoot(linalg::MakeVec(h_base_weight));
  }

  // Apply root split
  for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    auto p_tree = trees.at(fold_idx);
    auto candidate = root_entries.at(fold_idx);

    // TODO(jiamingy): Support learning rate.
    // TODO(jiamingy): Avoid device to host copies.
    std::vector<float> h_base_weight(candidate.base_weight.size());
    std::vector<float> h_left_weight(candidate.left_weight.size());
    std::vector<float> h_right_weight(candidate.right_weight.size());
    dh::CopyDeviceSpanToVector(&h_base_weight, candidate.base_weight);
    dh::CopyDeviceSpanToVector(&h_left_weight, candidate.left_weight);
    dh::CopyDeviceSpanToVector(&h_right_weight, candidate.right_weight);

    p_tree->ExpandNode(candidate.nidx, candidate.split.findex, candidate.split.fvalue,
                       candidate.split.dir == tree::kLeftDir, linalg::MakeVec(h_base_weight),
                       linalg::MakeVec(h_left_weight), linalg::MakeVec(h_right_weight));

    evaluators.at(fold_idx)->ApplyTreeSplit(ctx, p_tree, candidate);
  }
}
}  // namespace xgboost::cv
