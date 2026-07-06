/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <thrust/scatter.h>    // for scatter
#include <thrust/transform.h>  // for transform

#include <cstddef>  // for size_t
#include <limits>   // for numeric_limits
#include <memory>   // for make_unique
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/device_helpers.cuh"  // for ToSpan, tbegin, tcbegin, MakeIndexTransformIter
#include "../common/hist_util.h"         // for HistogramCuts
#include "../data/ellpack_page.cuh"      // for EllpackPageImpl
#include "../data/ellpack_page.h"        // for EllpackPage
#include "updater_gpu_common.cuh"        // for GPUTrainingParam
#include "updater_gpu_cv_hist.cuh"
#include "updater_gpu_hist.cuh"  // for CalcRootSum

namespace xgboost::tree::cuda_impl {
void CalcRootSumFolds(Context const* ctx,
                      std::vector<linalg::MatrixView<GradientPairInt64>> d_gpair,
                      std::vector<common::Span<GradientPairInt64>> root_sum) {
  auto k_folds = d_gpair.size();
  CHECK_EQ(k_folds, root_sum.size());
  for (std::size_t k = 0; k < k_folds; ++k) {
    CalcRootSum(ctx, d_gpair[k], root_sum[k]);
  }
}

FusedCvHistTreeMaker::FusedCvHistTreeMaker(Context const* ctx, TrainParam param,
                                           HistMakerTrainParam const* hist_param,
                                           std::shared_ptr<common::ColumnSampler> column_sampler,
                                           std::vector<bst_idx_t> batch_ptr,
                                           std::shared_ptr<common::HistogramCuts const> cuts,
                                           bool dense_compressed, std::size_t k_folds)
    : ctx_{ctx},
      param_{std::move(param)},
      hist_param_{hist_param},
      cuts_{std::move(cuts)},
      feature_groups_{std::make_unique<FeatureGroups>(*cuts_, dense_compressed,
                                                      DftMtHistShmemBytes(ctx->Ordinal()))},
      dense_compressed_{dense_compressed},
      batch_ptr_{std::move(batch_ptr)},
      k_folds_{k_folds},
      column_sampler_{std::move(column_sampler)} {
  CHECK_GT(k_folds_, 0);
  CHECK_GE(batch_ptr_.size(), 2);
  CHECK(column_sampler_);
  for (std::size_t k = 0; k < k_folds_; ++k) {
    folds_.emplace_back(std::make_unique<CvFoldDeviceState>());
  }
}

void FusedCvHistTreeMaker::Reset(DMatrix* p_fmat, cv::FoldInfoBatches const& finfo,
                                 cv::FoldGpairs const& gpairs) {
  auto const& info = p_fmat->Info();
  CHECK_EQ(finfo.KFolds(), k_folds_);
  CHECK_EQ(gpairs.gpairs.size(), k_folds_);
  auto n_batches = finfo.Size();
  CHECK_EQ(n_batches, batch_ptr_.size() - 1);

  auto n_rows = info.num_row_;
  CHECK_LE(n_rows, std::numeric_limits<RowIndexT>::max())
      << "Fused CV requires the row count to fit in 32 bits.";

  auto device = ctx_->Device();
  info.feature_types.SetDevice(device);
  cuts_->SetDevice(device);
  column_sampler_->Init(ctx_, info.num_col_, info.feature_weights, param_.colsample_bynode,
                        param_.colsample_bylevel, param_.colsample_bytree);

  // Number of targets is taken from the fold gradients. Single-target is n_targets == 1.
  n_targets_ = static_cast<bst_target_t>(gpairs.gpairs.at(0).Shape(1));
  CHECK_GE(n_targets_, 1);
  auto n_targets = static_cast<std::size_t>(n_targets_);

  for (std::size_t k = 0; k < k_folds_; ++k) {
    auto& fold = *folds_[k];
    CHECK_EQ(gpairs.gpairs[k].Shape(1), n_targets_) << "Inconsistent target count across folds.";
    auto fold_size = gpairs.gpairs[k].Shape(0);
    CHECK_EQ(fold_size, finfo.FoldSize(k));

    // 1. Build the global training row-index array grid_k (batch-concatenated).
    fold.grid.resize(fold_size);
    auto d_grid = dh::ToSpan(fold.grid);
    std::vector<common::Span<RowIndexT const>> batch_ridx(n_batches);
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < n_batches; ++i) {
      auto d_local = finfo.batches[i].TrainingFold(k);  // batch-local indices
      auto n_i = d_local.size();
      auto base = batch_ptr_[i];
      auto out = d_grid.subspan(cursor, n_i);
      thrust::transform(ctx_->CUDACtx()->CTP(), dh::tcbegin(d_local), dh::tcend(d_local),
                        dh::tbegin(out),
                        [=] __device__(bst_idx_t r) { return static_cast<RowIndexT>(r + base); });
      batch_ridx[i] = common::Span<RowIndexT const>{out.data(), out.size()};
      cursor += n_i;
    }
    CHECK_EQ(cursor, fold_size);

    // 2. Quantise the compact fold gradient (n_targets wide).
    fold.quantiser =
        std::make_unique<GradientQuantiserGroup>(ctx_, gpairs.gpairs[k].View(device), info);
    linalg::Matrix<GradientPairInt64> tmp_quant;
    CalcQuantizedGpairs(ctx_, gpairs.gpairs[k].View(device), fold.quantiser->DeviceSpan(),
                        &tmp_quant);

    // 3. Scatter the compact (fold_size x n_targets) gradient block into the global
    //    (N x n_targets) buffer; validation rows stay zero. Both buffers are column-major
    //    (F-contiguous), as required by the multi-target histogram build and produced by
    //    CalcQuantizedGpairs. In column-major layout element (row, target) lives at
    //    target * n_rows + row, so the scatter map turns a flat source index e (target
    //    e / fold_size, local row e % fold_size) into target * n_rows + grid[row].
    fold.d_gpair = linalg::Matrix<GradientPairInt64>{
        {n_rows, static_cast<bst_idx_t>(n_targets)}, device, linalg::kF};
    fold.d_gpair.Data()->Fill(GradientPairInt64{});
    auto src = tmp_quant.Data()->ConstDeviceSpan();
    auto dst = fold.d_gpair.Data()->DeviceSpan();
    auto const* d_grid_ptr = d_grid.data();
    auto n_rows_v = static_cast<std::size_t>(n_rows);
    auto map_it = dh::MakeIndexTransformIter([=] __device__(std::size_t e) -> std::size_t {
      auto t = e / fold_size;
      auto row = e % fold_size;
      return t * n_rows_v + static_cast<std::size_t>(d_grid_ptr[row]);
    });
    thrust::scatter(ctx_->CUDACtx()->CTP(), dh::tcbegin(src), dh::tcend(src), map_it,
                    dh::tbegin(dst));

    // 4. Seed the per-batch partitioners with the fold's global training rows.
    fold.partitioners.Reset(ctx_, batch_ridx);

    // 5. Reset the histogram builder. Each node histogram is TotalBins * n_targets wide.
    fold.histogram.Reset(ctx_, hist_param_->MaxCachedHistNodes(device),
                         cuts_->TotalBins() * static_cast<bst_idx_t>(n_targets),
                         /*force_global_memory=*/false);
  }
}

MultiEvaluateSplitSharedInputs FusedCvHistTreeMaker::MakeSharedInputs(
    std::size_t k, bst_feature_t max_active_feature) {
  return MultiEvaluateSplitSharedInputs{folds_[k]->quantiser->DeviceSpan(),
                                        cuts_->cut_ptrs_.ConstDeviceSpan(),
                                        cuts_->cut_values_.ConstDevicePointer(),
                                        {},  // feature_types, not implemented yet.
                                        cuts_->TotalBins(),
                                        max_active_feature,
                                        GPUTrainingParam{param_}};
}

void FusedCvHistTreeMaker::BuildRootHist(DMatrix* p_fmat) {
  auto device = ctx_->Device();
  for (auto const& fold : folds_) {
    CHECK_EQ(fold->partitioners.Size(), batch_ptr_.size() - 1);
  }

  std::int32_t i = 0;  // source batch index
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
    auto acc = page.Impl()->GetDeviceEllpack(ctx_, {});
    auto fg = feature_groups_->DeviceAccessor(device);
    for (std::size_t k = 0; k < k_folds_; ++k) {
      auto& fold = *folds_[k];
      auto ridx = fold.partitioners.At(i)->GetRows(RegTree::kRoot);
      if (ridx.empty()) {
        // The fold's validation block covers this whole batch.
        continue;
      }
      // Multi-target histogram build for the single root node.
      std::vector<common::Span<RowIndexT const>> h_ridxs{ridx};
      std::vector<common::Span<GradientPairInt64>> h_hists{
          fold.histogram.GetNodeHistogram(RegTree::kRoot)};
      std::vector<std::size_t> h_sizes_csum{0, ridx.size()};
      dh::device_vector<common::Span<RowIndexT const>> ridxs{h_ridxs};
      dh::device_vector<common::Span<GradientPairInt64>> hists{h_hists};
      fold.histogram.BuildHistogram(ctx_, acc, fg, fold.d_gpair.View(device), dh::ToSpan(ridxs),
                                    dh::ToSpan(hists), h_sizes_csum);
    }
    ++i;
  }
  CHECK_EQ(static_cast<std::size_t>(i), batch_ptr_.size() - 1);

  for (auto const& fold : folds_) {
    fold->histogram.AllReduceHist(ctx_, p_fmat->Info(), RegTree::kRoot, 1);
  }
}

MultiExpandEntry FusedCvHistTreeMaker::EvaluateRoot(DMatrix const* p_fmat, std::size_t k,
                                                    RegTree* p_tree) {
  auto& fold = *folds_[k];
  constexpr bst_node_t kRoot = RegTree::kRoot;
  auto device = ctx_->Device();

  // Compute the root leaf weight per target directly from the (already computed) root sum.
  // This is robust even when no valid split exists (the multi evaluator only fills the
  // node weight buffer when a split is found).
  std::vector<GradientPairInt64> h_root_sum(n_targets_);
  dh::CopyDeviceSpanToVector(&h_root_sum, dh::ToSpan(fold.root_sum));
  std::vector<float> h_weight(n_targets_);
  double sum_hess = 0;
  for (bst_target_t t = 0; t < n_targets_; ++t) {
    auto fp = (*fold.quantiser)[t].ToFloatingPoint(h_root_sum[t]);
    h_weight[t] = param_.learning_rate * CalcWeight(param_, fp);
    sum_hess += fp.GetHess();
  }

  // Evaluate the root split.
  auto sampled_features = column_sampler_->GetFeatureSet(ctx_, 0);
  sampled_features->SetDevice(device);
  auto feature_set = sampled_features->ConstDeviceSpan();
  MultiEvaluateSplitInputs input{kRoot, 0, dh::ToSpan(fold.root_sum), feature_set,
                                 fold.histogram.GetNodeHistogram(kRoot)};
  auto shared_inputs = this->MakeSharedInputs(k, static_cast<bst_feature_t>(feature_set.size()));
  auto entry = fold.evaluator.EvaluateSingleSplit(ctx_, input, shared_inputs);

  // Write the root leaf. XGBoost keeps two tree representations, so adapt to whichever the
  // caller provided; n_targets == 1 is stored as a scalar leaf.
  p_tree->SetRoot(linalg::MakeVec(h_weight), static_cast<float>(sum_hess));
  return entry;
}

std::vector<MultiExpandEntry> FusedCvHistTreeMaker::InitRoots(DMatrix* p_fmat,
                                                              std::vector<RegTree*> const& trees) {
  CHECK_EQ(trees.size(), k_folds_);

  // 1. Allocate the root histogram for every fold.
  for (auto const& fold : folds_) {
    fold->histogram.AllocateHistograms(ctx_, {RegTree::kRoot});
  }

  // 2. Root sum (n_targets wide) for every fold.
  std::vector<linalg::MatrixView<GradientPairInt64>> views;
  std::vector<common::Span<GradientPairInt64>> sums;
  for (std::size_t k = 0; k < k_folds_; ++k) {
    folds_[k]->root_sum.resize(n_targets_);
    views.push_back(folds_[k]->d_gpair.View(ctx_->Device()));
    sums.push_back(dh::ToSpan(folds_[k]->root_sum));
  }
  CalcRootSumFolds(ctx_, views, sums);

  // 3. Root histograms (the fusion: each source page fetched once, reused by all folds).
  this->BuildRootHist(p_fmat);

  // 4. Evaluate the root split for every fold.
  std::vector<MultiExpandEntry> entries(k_folds_);
  for (std::size_t k = 0; k < k_folds_; ++k) {
    entries[k] = this->EvaluateRoot(p_fmat, k, trees[k]);
  }
  return entries;
}

common::Span<GradientPairInt64 const> FusedCvHistTreeMaker::RootHistogram(std::size_t k) {
  return folds_.at(k)->histogram.GetNodeHistogram(RegTree::kRoot);
}

RowPartitionerBatches& FusedCvHistTreeMaker::Partitioners(std::size_t k) {
  return folds_.at(k)->partitioners;
}

GradientPairPrecise FusedCvHistTreeMaker::RootSum(std::size_t k, bst_target_t t) const {
  auto const& fold = *folds_.at(k);
  GradientPairInt64 rs{};
  dh::safe_cuda(cudaMemcpyAsync(&rs, fold.root_sum.data() + t, sizeof(GradientPairInt64),
                                cudaMemcpyDeviceToHost, ctx_->CUDACtx()->Stream()));
  ctx_->CUDACtx()->Stream().Sync();
  return (*fold.quantiser)[t].ToFloatingPoint(rs);
}

GradientQuantiser const& FusedCvHistTreeMaker::Quantiser(std::size_t k, bst_target_t t) const {
  return (*folds_.at(k)->quantiser)[t];
}
}  // namespace xgboost::tree::cuda_impl
