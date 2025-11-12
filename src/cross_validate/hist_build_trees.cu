/**
 * Copyright 2025, XGBoost contributors
 */
#include <vector>

#include "../common/cuda_context.cuh"
#include "../data/batch_utils.h"
#include "../data/ellpack_page.cuh"
#include "folds.h"
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
void BuildTrees(Context const* ctx, DMatrix* p_fmat, std::vector<GradientContainer const*> gpairs,
                std::vector<std::vector<std::vector<cv::Segment>>> const& segments,
                std::vector<RegTree*> trees) {
  // len(trees) == n_folds

  // Build histogram for the root nodes
  std::int32_t batch_idx = 0;
  auto n_folds = trees.size();
  CHECK_EQ(segments.size(), p_fmat->NumBatches());
  dh::device_vector<GradientPairInt64> node_histogam;  // fixme
  auto d_node_histogram = dh::ToSpan(node_histogam);

  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
    page.Impl()->Visit(ctx, {}, [&](auto&& d_acc) {
      // d_acc is either EllpackDeviceAccessor or DoubleEllpackAccessor
      auto const& batch_segments = segments.at(batch_idx);
      CHECK_EQ(batch_segments.size(), n_folds);
      for (std::size_t fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
        auto fold_grad = gpairs[fold_idx];
        // fixme: split-grad
        auto d_grad = fold_grad->FullGradOnly()->View(ctx->Device());
        auto const& h_fold_segments = batch_segments[fold_idx];
        dh::device_vector<Segment> fold_segments(h_fold_segments);
        auto d_fold_segments = dh::ToSpan(fold_segments);
        auto n_rows = h_fold_segments.back().End();  // fixme: sort
        dh::device_vector<bst_idx_t> train_idx;      // fixme: regen
        CHECK_EQ(train_idx.size(), n_rows);
        auto d_train_idx = dh::ToSpan(train_idx);
        // dh::LaunchN(n_rows * d_acc.row_stride, ctx->CUDACtx()->Stream(),
        //             [=] XGBOOST_DEVICE(std::size_t i) {
        //               auto iidx = i / d_acc.row_stride;
        //               auto ridx = d_train_idx[iidx];
        //               auto fidx = i % d_acc.row_stride;

        //               auto eidx = ridx * d_acc.row_stride + fidx;
        //               auto compressed_bin = d_acc.gidx_iter[eidx];
        //               // fixme: target
        //               auto gpair = d_grad(ridx, 0);
        //               // quantizer
        //               // d_node_histogram[compressed_bin] += gpair;
        //             });
      }
    });
    batch_idx++;
  }

  // Evaluate roots
}

void BuildTrees(Context const* ctx, DMatrix* p_fmat,
                std::vector<std::vector<std::unique_ptr<GradientContainer>>> const& gpairs,
                std::vector<std::vector<std::vector<bst_idx_t>>> const& tr_idx,
                std::vector<RegTree*> trees) {
  std::int32_t batch_idx = 0;
  auto n_folds = trees.size();
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
    // init root
  }
}
}  // namespace xgboost::cv
