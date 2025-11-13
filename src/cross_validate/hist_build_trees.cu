/**
 * Copyright 2025, XGBoost contributors
 */
#include <vector>

#include "../common/cuda_context.cuh"
#include "../common/linalg_iter.h"
#include "../data/batch_utils.h"
#include "../data/ellpack_page.cuh"
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

// Maybe we can modify the multi-target builder to handle many trees
void BuildTrees(Context const* ctx, DMatrix* p_fmat,
                std::vector<std::vector<std::unique_ptr<GradientContainer>>> const& gpairs,
                std::vector<std::vector<std::vector<bst_idx_t>>> const& tr_idx,
                std::vector<RegTree*> trees) {
  std::int32_t batch_idx = 0;
  auto n_folds = trees.size();
  std::int32_t k = 0;
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
    // init root
    auto const& batch_gpairs = gpairs.at(k);
    auto const& batch_tr_idx = tr_idx.at(k);

    for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto d_gpair = batch_gpairs[0]->gpair.View(ctx->Device());
      auto const& fold_tr_idx = batch_tr_idx.at(fold_idx);
      dh::device_vector<bst_idx_t> d_fold_tr_idx{fold_tr_idx};
      auto per_d_gpair = linalg::PermutationTensorView{d_gpair, dh::ToSpan(d_fold_tr_idx)};
      auto root_sum = tree::cuda_impl::CalcRootSum(ctx, d_gpair, {});  // fixme
    }

    ++k;
  }
}
}  // namespace xgboost::cv
