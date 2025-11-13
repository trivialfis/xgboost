/**
 * Copyright 2025, XGBoost contributors
 */
#include <vector>

#include "../common/cuda_context.cuh"
#include "../common/deterministic.cuh"
#include "../common/linalg_iter.h"
#include "../common/linalg_op.cuh"
#include "../data/batch_utils.h"
#include "../data/ellpack_page.cuh"
#include "../tree/gpu_hist/quantiser.cuh"
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
  std::vector<Pair> running_sum(n_folds);
  std::vector<bst_idx_t> running_sum_rows(n_folds);
  for (std::size_t batch_idx = 0; batch_idx < p_fmat->NumBatches(); ++batch_idx) {
    auto const& batch_gpairs = gpairs.at(batch_idx);
    for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto fold_gpair = batch_gpairs.at(fold_idx)->gpair.View(ctx->Device());
      auto beg = thrust::make_transform_iterator(linalg::tcbegin(fold_gpair), Clip());
      running_sum[fold_idx] = dh::Reduce(ctx->CUDACtx()->CTP(), beg, beg + fold_gpair.Size(),
                                         running_sum[fold_idx], thrust::plus<Pair>{});
      running_sum_rows[fold_idx] += fold_gpair.Shape<0>();
    }
  }
  for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    split_quantizer.emplace_back(
        CreateQuantizer(running_sum[fold_idx], running_sum_rows[fold_idx]));
  }

  // Accumulate the root sum from all batches
  // Init root
  std::int32_t batch_idx = 0;
  dh::device_vector<GradientPairInt64> root_sums(n_folds * n_targets);
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
    auto const& batch_gpairs = gpairs.at(batch_idx);
    auto const& batch_tr_idx = tr_idx.at(batch_idx);

    for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto d_gpair = batch_gpairs[0]->gpair.View(ctx->Device());
      // We can use d_gpair without permutation indexing as it's calculated from the fold.
      auto fold_root_sum = dh::ToSpan(root_sums).subspan(fold_idx * n_targets, n_targets);
      // fixme: multi
      dh::device_vector<tree::GradientQuantiser> d_q{*split_quantizer.at(fold_idx)};
      tree::cuda_impl::CalcRootSum(ctx, d_gpair, dh::ToSpan(d_q), fold_root_sum);
    }

    ++batch_idx;
  }

  // Build root histogram.
}
}  // namespace xgboost::cv
