/**
 * Copyright 2026, XGBoost Contributors
 */
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/sequence.h>                     // for sequence

#include <algorithm>    // for copy_if, max
#include <cmath>        // for isnan
#include <cstdint>      // for int32_t
#include <memory>       // for make_unique
#include <type_traits>  // for remove_reference_t
#include <utility>      // for move
#include <vector>       // for vector

#include "../collective/aggregator.h"         // for GlobalSum
#include "../collective/communicator-inl.h"   // for IsDistributed
#include "../common/categorical.h"            // for KCatBitField, Decision, GetNodeCats
#include "../common/cuda_context.cuh"         // for CUDAContext
#include "../common/cuda_rt_utils.h"          // for SetDevice
#include "../common/cuda_stream.h"            // for DefaultStream
#include "../common/device_helpers.cuh"       // for Reduce, ToSpan, LaunchN, CopyTo
#include "../common/device_vector.cuh"        // for DeviceUVector
#include "../common/linalg_op.cuh"            // for tcbegin
#include "../data/batch_utils.h"              // for StaticBatch
#include "../data/ellpack_page.cuh"           // for EllpackPageImpl
#include "../data/ellpack_page.h"             // for EllpackPage
#include "param.h"                            // for CalcWeight, GPUTrainingParam
#include "sample_position.h"                  // for SamplePosition
#include "updater_gpu_common.cuh"             // for DeviceSplitCandidate
#include "updater_gpu_hist_cv.cuh"            // for GPUFusedCVHistMaker
#include "xgboost/collective/result.h"        // for SafeColl, Success
#include "xgboost/linalg.h"                   // for MakeVec, MakeTensorView

namespace xgboost::tree {
using xgboost::cuda_impl::StaticBatch;

namespace {
// Per-node data passed to the partition kernel. Mirrors the (translation-unit-local)
// struct in `updater_gpu_hist.cu`; redefined here because the fused maker does not reuse
// `GPUHistMakerDevice`.
struct CVNodeSplitData {
  RegTree::Node split_node;
  FeatureType split_type;
  common::KCatBitField node_cats;
};

template <typename Accessor>
struct CVGoLeftOp {
  Accessor d_matrix;

  __device__ bool operator()(cuda_impl::RowIndexT ridx, CVNodeSplitData const& data) const {
    RegTree::Node const& node = data.split_node;
    float cut_value = d_matrix.GetFvalue(ridx, node.SplitIndex());
    bool go_left = true;
    if (isnan(cut_value)) {
      go_left = node.DefaultLeft();
    } else {
      if (data.split_type == FeatureType::kCategorical) {
        go_left = common::Decision(data.node_cats.Bits(), cut_value);
      } else {
        go_left = cut_value <= node.SplitCond();
      }
    }
    return go_left;
  }
};

struct CVPartitionNodes {
  std::vector<bst_node_t> nidx;
  std::vector<bst_node_t> left_nidx;
  std::vector<bst_node_t> right_nidx;
  std::vector<CVNodeSplitData> split_data;

  CVPartitionNodes() = default;
  explicit CVPartitionNodes(std::size_t n)
      : nidx(n), left_nidx(n), right_nidx(n), split_data(n) {}
};

// Traverse one tree (numerical splits) for row `ridx` over the binned Ellpack accessor and
// return the reached leaf value. `nodes` is the tree's node array with node 0 as the root.
// Shared by the fused validation predictor and the standalone `PredictTreeBinned` so a
// reference path predicts bit-identical trees identically (review #2 R2-D).
template <typename Accessor>
__device__ float BinnedLeaf(Accessor const& acc, bst_idx_t ridx,
                            common::Span<RegTree::Node const> nodes) {
  RegTree::Node node = nodes[0];
  while (!node.IsLeaf()) {
    float fvalue = acc.GetFvalue(ridx, node.SplitIndex());
    bool go_left = isnan(fvalue) ? node.DefaultLeft() : (fvalue <= node.SplitCond());
    node = nodes[go_left ? node.LeftChild() : node.RightChild()];
  }
  return node.LeafValue();
}

// Device functor for the fused validation predictor: route each row to the fold whose
// validation block contains it and accumulate that fold's tree leaf into its margin. Defined
// as a functor (not a `__device__` lambda) because the accessor is obtained via a generic
// `Visit` lambda, inside which extended device lambdas are not allowed.
template <typename Accessor>
struct CVPredictValidOp {
  Accessor acc;
  bst_idx_t base;
  common::Span<RegTree::Node const> nodes;
  common::Span<std::uint32_t const> offset;
  common::Span<bst_idx_t const> valid_ptr;
  common::Span<float*> margin;
  std::int32_t n_folds;

  __device__ void operator()(std::size_t i) const {
    bst_idx_t ridx = base + i;
    std::int32_t f = 0;
    while (f + 1 < n_folds && ridx >= valid_ptr[f + 1]) {
      ++f;
    }
    common::Span<RegTree::Node const> fold_nodes{nodes.data() + offset[f],
                                                 offset[f + 1] - offset[f]};
    margin[f][ridx] += BinnedLeaf(acc, ridx, fold_nodes);
  }
};

// Device functor adding a single tree's leaf to a local-indexed margin over every row.
template <typename Accessor>
struct PredictTreeBinnedOp {
  Accessor acc;
  bst_idx_t base;
  common::Span<RegTree::Node const> nodes;
  common::Span<float> out;

  __device__ void operator()(std::size_t i) const {
    bst_idx_t ridx = base + i;
    out[ridx] += BinnedLeaf(acc, ridx, nodes);
  }
};

// Build a device array of the fold's global training-row indices for one source batch by
// concatenating the (at most two) contiguous runs. No Ellpack data is copied — these are
// just the row indices used to seed the zero-copy logical slice.
dh::DeviceUVector<bst_idx_t> BuildBatchRidx(Context const* ctx,
                                            std::vector<RowRange> const& runs) {
  bst_idx_t n = 0;
  for (auto const& r : runs) {
    n += r.second - r.first;
  }
  dh::DeviceUVector<bst_idx_t> out;
  out.resize(n);
  auto cuctx = ctx->CUDACtx();
  auto* p = out.data();
  bst_idx_t off = 0;
  for (auto const& r : runs) {
    auto len = r.second - r.first;
    thrust::sequence(cuctx->CTP(), thrust::device_pointer_cast(p + off),
                     thrust::device_pointer_cast(p + off + len), static_cast<bst_idx_t>(r.first));
    off += len;
  }
  return out;
}
}  // anonymous namespace

GPUFusedCVHistMaker::GPUFusedCVHistMaker(Context const* ctx, TrainParam param,
                                         HistMakerTrainParam const* hist_param, CVFoldInfo folds,
                                         std::vector<bst_idx_t> batch_ptr,
                                         std::shared_ptr<common::HistogramCuts const> cuts,
                                         bool dense_compressed, bst_feature_t n_features)
    : ctx_{ctx},
      param_{std::move(param)},
      hist_param_{hist_param},
      folds_{std::move(folds)},
      batch_ptr_{std::move(batch_ptr)},
      cuts_{std::move(cuts)},
      dense_compressed_{dense_compressed},
      n_features_{n_features},
      feature_groups_{std::make_unique<FeatureGroups>(*cuts_, dense_compressed_,
                                                      DftStHistShmemBytes(ctx_->Ordinal()))} {
  fold_.reserve(folds_.n_folds);
  for (std::int32_t f = 0; f < folds_.n_folds; ++f) {
    fold_.emplace_back(std::make_unique<FoldDeviceState>(ctx_, param_, n_features_));
  }
}

void GPUFusedCVHistMaker::ResetFold(FoldDeviceState* st, MetaInfo const& info,
                                    HostDeviceVector<GradientPair> const* gpair) {
  // Quantise the gradient using the fold's training-row count so the rounding factor — and
  // therefore the quantised gradients — match a baseline trained on just the fold's rows
  // (review #2 R2-A). The global-sized gpair has validation rows zeroed, so the clipped sum
  // is unchanged.
  st->quantiser = std::make_unique<GradientQuantiserGroup>(
      ctx_, linalg::MakeVec(ctx_->Device(), gpair->ConstDeviceSpan()), info, st->fold_rows);
  auto gpair_view = linalg::MakeTensorView(ctx_, gpair->ConstDeviceSpan(), gpair->Size(), 1);
  CalcQuantizedGpairs(ctx_, gpair_view, st->quantiser->DeviceSpan(), &st->d_gpair);

  // Seed the per-batch partitioners with this fold's training-row indices over the shared
  // pages (zero-copy logical slice). Inactive batches get an empty row list.
  std::size_t n_batches = batch_ptr_.size() - 1;
  std::vector<dh::DeviceUVector<bst_idx_t>> storage;
  storage.reserve(n_batches);
  std::vector<common::Span<bst_idx_t const>> batch_ridx;
  batch_ridx.reserve(n_batches);
  for (std::size_t k = 0; k < n_batches; ++k) {
    storage.emplace_back(BuildBatchRidx(ctx_, st->view.per_batch_runs[k]));
    batch_ridx.emplace_back(storage.back().data(), storage.back().size());
  }
  st->partitioners.Reset(ctx_, batch_ridx);

  // Initialize the evaluator / column sampler / constraints exactly like the single-fold
  // path so the grown tree is bit-identical to the baseline.
  st->column_sampler->Init(ctx_, info.num_col_, info.feature_weights, param_.colsample_bynode,
                           param_.colsample_bylevel, param_.colsample_bytree);
  st->interaction_constraints.Reset(ctx_);
  st->evaluator.Reset(ctx_, *cuts_, info.feature_types.ConstDeviceSpan(), info.num_col_, param_,
                      info.IsColumnSplit());

  st->histogram.Reset(ctx_, hist_param_->MaxCachedHistNodes(ctx_->Device()), cuts_->TotalBins(),
                      false);
}

GradientPairInt64 GPUFusedCVHistMaker::RootSum(FoldDeviceState* st, MetaInfo const& info) const {
  auto gpair_it = linalg::tcbegin(st->d_gpair.View(ctx_->Device()));
  // The validation entries are zero, so summing over the whole global buffer yields the same
  // (integer, order-independent) sum as a baseline over just the fold's rows.
  GradientPairInt64 root_sum =
      dh::Reduce(ctx_->CUDACtx()->CTP(), gpair_it, gpair_it + st->d_gpair.Size(),
                 GradientPairInt64{}, cuda::std::plus<GradientPairInt64>{});
  using ReduceT = typename GradientPairInt64::ValueT;
  auto rc = collective::GlobalSum(ctx_, info,
                                  linalg::MakeVec(reinterpret_cast<ReduceT*>(&root_sum), 2));
  collective::SafeColl(rc);
  return root_sum;
}

void GPUFusedCVHistMaker::BuildHist(FoldDeviceState* st, EllpackPage const& page, std::int32_t k,
                                    bst_node_t nidx) {
  auto d_ridx = st->partitioners.At(k)->GetRows(nidx);
  if (d_ridx.empty()) {
    // Node has no rows in this batch — can happen with external memory when all of a node's
    // rows go to its sibling, or when a fold's training rows do not cover this page.
    return;
  }
  auto d_node_hist = st->histogram.GetNodeHistogram(nidx);
  auto acc = page.Impl()->GetDeviceEllpack(ctx_, {});
  auto gpair = st->d_gpair.View(ctx_->Device());
  st->histogram.BuildHistogram(ctx_, acc, feature_groups_->DeviceAccessor(ctx_->Device()),
                               gpair.Values(), d_ridx, d_node_hist);
}

GPUExpandEntry GPUFusedCVHistMaker::EvaluateRootSplit(FoldDeviceState* st, DMatrix const* p_fmat,
                                                      GradientPairInt64 root_sum) {
  bst_node_t nidx = RegTree::kRoot;
  GPUTrainingParam gpu_param(param_);
  auto sampled_features = st->column_sampler->GetFeatureSet(ctx_, 0);
  sampled_features->SetDevice(ctx_->Device());
  common::Span<bst_feature_t const> feature_set =
      st->interaction_constraints.Query(sampled_features->ConstDeviceSpan(), nidx);
  EvaluateSplitInputs inputs{nidx, 0, root_sum, feature_set,
                             st->histogram.GetNodeHistogram(nidx)};
  EvaluateSplitSharedInputs shared_inputs{gpu_param,
                                          (*st->quantiser)[0],
                                          p_fmat->Info().feature_types.ConstDeviceSpan(),
                                          cuts_->cut_ptrs_.ConstDeviceSpan(),
                                          cuts_->cut_values_.ConstDeviceSpan(),
                                          p_fmat->IsDense() && !collective::IsDistributed()};
  return st->evaluator.EvaluateSingleSplit(ctx_, inputs, shared_inputs);
}

void GPUFusedCVHistMaker::EvaluateSplits(FoldDeviceState* st, DMatrix const* p_fmat,
                                         std::vector<GPUExpandEntry> const& candidates,
                                         common::Span<GPUExpandEntry> out) {
  if (candidates.empty()) {
    return;
  }
  dh::TemporaryArray<EvaluateSplitInputs> d_node_inputs(2 * candidates.size());
  std::vector<bst_node_t> nidx(2 * candidates.size());
  auto h_node_inputs = pinned2_.GetSpan<EvaluateSplitInputs>(2 * candidates.size());
  EvaluateSplitSharedInputs shared_inputs{GPUTrainingParam{param_},
                                          (*st->quantiser)[0],
                                          p_fmat->Info().feature_types.ConstDeviceSpan(),
                                          cuts_->cut_ptrs_.ConstDeviceSpan(),
                                          cuts_->cut_values_.ConstDeviceSpan(),
                                          p_fmat->IsDense() && !collective::IsDistributed()};
  dh::TemporaryArray<GPUExpandEntry> entries(2 * candidates.size());
  std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> feature_sets;
  auto sc_tree = st->tree->HostScView();
  for (std::size_t i = 0; i < candidates.size(); i++) {
    auto candidate = candidates.at(i);
    bst_node_t left_nidx = sc_tree.LeftChild(candidate.nidx);
    bst_node_t right_nidx = sc_tree.RightChild(candidate.nidx);
    nidx[i * 2] = left_nidx;
    nidx[i * 2 + 1] = right_nidx;
    auto left_sampled_features = st->column_sampler->GetFeatureSet(ctx_, st->tree->GetDepth(left_nidx));
    feature_sets.emplace_back(left_sampled_features);
    common::Span<bst_feature_t const> left_feature_set =
        st->interaction_constraints.Query(left_sampled_features->ConstDeviceSpan(), left_nidx);
    auto right_sampled_features =
        st->column_sampler->GetFeatureSet(ctx_, st->tree->GetDepth(right_nidx));
    feature_sets.emplace_back(right_sampled_features);
    common::Span<bst_feature_t const> right_feature_set =
        st->interaction_constraints.Query(right_sampled_features->ConstDeviceSpan(), right_nidx);
    h_node_inputs[i * 2] = {left_nidx, candidate.depth + 1, candidate.split.left_sum,
                            left_feature_set, st->histogram.GetNodeHistogram(left_nidx)};
    h_node_inputs[i * 2 + 1] = {right_nidx, candidate.depth + 1, candidate.split.right_sum,
                                right_feature_set, st->histogram.GetNodeHistogram(right_nidx)};
  }
  bst_feature_t max_active_features = 0;
  for (auto input : h_node_inputs) {
    max_active_features =
        std::max(max_active_features, static_cast<bst_feature_t>(input.feature_set.size()));
  }
  dh::safe_cuda(cudaMemcpyAsync(d_node_inputs.data().get(), h_node_inputs.data(),
                                h_node_inputs.size() * sizeof(EvaluateSplitInputs),
                                cudaMemcpyDefault));
  st->evaluator.EvaluateSplits(ctx_, nidx, max_active_features, dh::ToSpan(d_node_inputs),
                               shared_inputs, dh::ToSpan(entries));
  dh::safe_cuda(cudaMemcpyAsync(out.data(), entries.data().get(),
                                sizeof(GPUExpandEntry) * entries.size(), cudaMemcpyDeviceToHost));
}

void GPUFusedCVHistMaker::ApplySplit(FoldDeviceState* st, GPUExpandEntry const& candidate) {
  RegTree& tree = *st->tree;
  auto base_weight = candidate.base_weight;
  auto left_weight = candidate.left_weight * param_.learning_rate;
  auto right_weight = candidate.right_weight * param_.learning_rate;
  auto const& q = (*st->quantiser)[0];
  auto parent_hess =
      q.ToFloatingPoint(candidate.split.left_sum + candidate.split.right_sum).GetHess();
  auto left_hess = q.ToFloatingPoint(candidate.split.left_sum).GetHess();
  auto right_hess = q.ToFloatingPoint(candidate.split.right_sum).GetHess();

  auto is_cat = candidate.split.is_cat;
  if (is_cat) {
    CHECK(common::CheckNAN(candidate.split.fvalue));
    std::vector<common::CatBitField::value_type> split_cats;
    auto h_cats = st->evaluator.GetHostNodeCats(candidate.nidx);
    auto n_bins_feature = cuts_->FeatureBins(candidate.split.findex);
    split_cats.resize(common::CatBitField::ComputeStorageSize(n_bins_feature), 0);
    CHECK_LE(split_cats.size(), h_cats.size());
    std::copy(h_cats.data(), h_cats.data() + split_cats.size(), split_cats.data());
    tree.ExpandCategorical(candidate.nidx, candidate.split.findex, split_cats,
                           candidate.split.dir == kLeftDir, base_weight, left_weight, right_weight,
                           candidate.split.loss_chg, parent_hess, left_hess, right_hess);
  } else {
    CHECK(!common::CheckNAN(candidate.split.fvalue));
    tree.ExpandNode(candidate.nidx, candidate.split.findex, candidate.split.fvalue,
                    candidate.split.dir == kLeftDir, base_weight, left_weight, right_weight,
                    candidate.split.loss_chg, parent_hess, left_hess, right_hess);
  }
  st->evaluator.ApplyTreeSplit(candidate, st->tree);

  auto const& parent = tree[candidate.nidx];
  st->interaction_constraints.Split(ctx_, candidate.nidx, parent.SplitIndex(), parent.LeftChild(),
                                    parent.RightChild());
}

void GPUFusedCVHistMaker::ReduceHist(FoldDeviceState* st, DMatrix* p_fmat, MetaInfo const& info) {
  if (st->valid_candidates.empty()) {
    return;
  }
  st->histogram.AllReduceHist(ctx_, info, st->build_nidx.at(0), st->build_nidx.size());
  auto need_build =
      st->histogram.SubtractHist(ctx_, st->valid_candidates, st->build_nidx, st->subtraction_nidx);
  if (need_build.empty()) {
    return;
  }
  // Slow path: a parent histogram was evicted from the bounded cache so subtraction is
  // impossible. For the POC the cache is sized so this does not happen for tested tree
  // sizes; when it does, rebuild over the shared pages (still no per-fold extra fetch beyond
  // this rebuild).
  std::int32_t k = 0;
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
    if (st->BatchActive(k)) {
      for (auto nidx : need_build) {
        this->BuildHist(st, page, k, nidx);
      }
    }
    ++k;
  }
  for (auto nidx : need_build) {
    st->histogram.AllReduceHist(ctx_, info, nidx, 1);
  }
}

void GPUFusedCVHistMaker::FinalisePosition(FoldDeviceState* st) {
  st->p_out_position->SetDevice(ctx_->Device());
  st->p_out_position->Resize(folds_.n_rows);
  auto d_out_position = st->p_out_position->DeviceSpan();
  auto gpair = st->d_gpair.View(ctx_->Device());
  for (std::int32_t k : st->view.active_batches) {
    auto& part = st->partitioners.At(k);
    auto base_ridx = batch_ptr_[k];
    auto n_samples = batch_ptr_.at(k + 1) - base_ridx;
    part->FinalisePosition(ctx_, d_out_position.subspan(base_ridx, n_samples), base_ridx,
                           cuda_impl::EncodeOp{gpair});
  }
}

void GPUFusedCVHistMaker::UpdateTrees(
    DMatrix* p_fmat, std::vector<HostDeviceVector<GradientPair>*> const& gpair,
    std::vector<RegTree*> const& trees,
    std::vector<HostDeviceVector<bst_node_t>*> const& positions) {
  curt::SetDevice(ctx_->Ordinal());
  auto& info = p_fmat->Info();
  info.feature_types.SetDevice(ctx_->Device());

  auto K = this->NumFolds();
  CHECK_EQ(static_cast<std::int32_t>(gpair.size()), K);
  CHECK_EQ(static_cast<std::int32_t>(trees.size()), K);
  CHECK_EQ(static_cast<std::int32_t>(positions.size()), K);
  CHECK_EQ(p_fmat->NumBatches(), batch_ptr_.size() - 1);

  // ---- Per-fold reset ----
  for (std::int32_t f = 0; f < K; ++f) {
    auto* st = fold_[f].get();
    st->tree = trees[f];
    st->p_out_position = positions[f];
    st->view = MakeFoldBatchView(folds_, f, batch_ptr_);
    st->fold_rows = st->view.TotalRows();
    gpair[f]->SetDevice(ctx_->Device());
    this->ResetFold(st, info, gpair[f]);
  }

  // ---- Root: one shared page pass to build all folds' root histograms ----
  for (std::int32_t f = 0; f < K; ++f) {
    fold_[f]->histogram.AllocateHistograms(ctx_, {RegTree::kRoot});
  }
  {
    std::int32_t k = 0;
    for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
      for (std::int32_t f = 0; f < K; ++f) {
        auto* st = fold_[f].get();
        if (st->BatchActive(k)) {
          this->BuildHist(st, page, k, RegTree::kRoot);
        }
      }
      ++k;
    }
  }
  for (std::int32_t f = 0; f < K; ++f) {
    auto* st = fold_[f].get();
    auto root_sum = this->RootSum(st, info);
    auto root_sum_fp = (*st->quantiser)[0].ToFloatingPoint(root_sum);
    st->tree->Stat(RegTree::kRoot).sum_hess = root_sum_fp.GetHess();
    auto weight = CalcWeight(param_, root_sum_fp);
    st->tree->Stat(RegTree::kRoot).base_weight = weight;
    (*st->tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    auto root_entry = this->EvaluateRootSplit(st, p_fmat, root_sum);
    st->driver.Push({root_entry});
    st->expand_set = st->driver.Pop();
  }

  // ---- Level loop ----
  auto any_expanding = [&]() {
    for (std::int32_t f = 0; f < K; ++f) {
      if (!fold_[f]->expand_set.empty()) {
        return true;
      }
    }
    return false;
  };

  while (any_expanding()) {
    std::vector<CVPartitionNodes> level_nodes(K);
    // 1. Apply splits, pick build/subtraction nodes, allocate histograms, and stage the
    //    partition node arrays. (All host-side, no page access.)
    for (std::int32_t f = 0; f < K; ++f) {
      auto* st = fold_[f].get();
      if (st->expand_set.empty()) {
        st->valid_candidates.clear();
        st->build_nidx.clear();
        st->subtraction_nidx.clear();
        continue;
      }
      for (auto const& candidate : st->expand_set) {
        this->ApplySplit(st, candidate);
      }
      st->valid_candidates.clear();
      std::copy_if(st->expand_set.begin(), st->expand_set.end(),
                   std::back_inserter(st->valid_candidates),
                   [&](auto const& e) { return st->driver.IsChildValid(e); });

      st->build_nidx.assign(st->valid_candidates.size(), 0);
      st->subtraction_nidx.assign(st->valid_candidates.size(), 0);
      auto sc_tree = st->tree->HostScView();
      cuda_impl::AssignNodes(sc_tree, st->valid_candidates, common::Span<bst_node_t>{st->build_nidx},
                             common::Span<bst_node_t>{st->subtraction_nidx},
                             [&](GPUExpandEntry const& e) {
                               auto const& q = (*st->quantiser)[0];
                               auto left_sum = q.ToFloatingPoint(e.split.left_sum);
                               auto right_sum = q.ToFloatingPoint(e.split.right_sum);
                               return right_sum.GetHess() < left_sum.GetHess();
                             });
      st->histogram.AllocateHistograms(ctx_, st->build_nidx, st->subtraction_nidx);

      // Partition all of `expand_set` (external-memory strategy) so positions for every node
      // are tracked directly by the partitioner and `FinalisePosition` needs no page re-read.
      auto& nodes = level_nodes[f];
      nodes = CVPartitionNodes(st->expand_set.size());
      auto sc = st->tree->HostScView();
      for (std::size_t i = 0, n = st->expand_set.size(); i < n; ++i) {
        auto const& e = st->expand_set[i];
        nodes.nidx[i] = e.nidx;
        nodes.left_nidx[i] = sc.LeftChild(e.nidx);
        nodes.right_nidx[i] = sc.RightChild(e.nidx);
        nodes.split_data[i] = CVNodeSplitData{sc.nodes[e.nidx], sc.SplitType(e.nidx),
                                              st->evaluator.GetDeviceNodeCats(e.nidx)};
      }
    }

    // 2. One shared page pass: partition + build histograms for every fold.
    {
      std::int32_t k = 0;
      for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
        for (std::int32_t f = 0; f < K; ++f) {
          auto* st = fold_[f].get();
          if (st->expand_set.empty() || !st->BatchActive(k)) {
            continue;
          }
          auto& nodes = level_nodes[f];
          page.Impl()->Visit(ctx_, {}, [&](auto&& d_acc) {
            using Acc = std::remove_reference_t<decltype(d_acc)>;
            CVGoLeftOp<Acc> go_left{d_acc};
            st->partitioners.UpdatePositionBatch(
                ctx_, k, nodes.nidx, nodes.left_nidx, nodes.right_nidx, nodes.split_data,
                cuda_impl::GoLeftWrapperOp<CVGoLeftOp<Acc>>{go_left});
          });
          for (auto nidx : st->build_nidx) {
            this->BuildHist(st, page, k, nidx);
          }
        }
        ++k;
      }
    }

    // 3. Per-fold reduce + evaluate + enqueue the next level.
    for (std::int32_t f = 0; f < K; ++f) {
      auto* st = fold_[f].get();
      if (st->expand_set.empty()) {
        continue;
      }
      this->ReduceHist(st, p_fmat, info);
      auto new_candidates =
          pinned_.GetSpan<GPUExpandEntry>(st->valid_candidates.size() * 2, GPUExpandEntry{});
      this->EvaluateSplits(st, p_fmat, st->valid_candidates, new_candidates);
      curt::DefaultStream().Sync();
      st->driver.Push(new_candidates.begin(), new_candidates.end());
      st->expand_set = st->driver.Pop();
    }
  }

  // ---- Finalize positions (no page read for external memory) ----
  for (std::int32_t f = 0; f < K; ++f) {
    this->FinalisePosition(fold_[f].get());
  }
}

void GPUFusedCVHistMaker::UpdatePredictionCache(std::int32_t f, linalg::MatrixView<float> out_preds,
                                                RegTree const* p_tree) {
  CHECK(p_tree);
  CHECK(out_preds.Device().IsCUDA());
  CHECK_EQ(out_preds.Shape(1), 1);
  auto* st = fold_[f].get();

  dh::CachingDeviceUVector<RegTree::Node> nodes;
  dh::CopyTo(p_tree->GetNodes(DeviceOrd::CPU()), &nodes, ctx_->CUDACtx()->Stream());
  common::Span<RegTree::Node> d_nodes = dh::ToSpan(nodes);

  auto d_position = st->p_out_position->ConstDeviceSpan();
  // Drive the update over the fold's TRAIN rows only (review #2 R2-B); validation slots of
  // the global position buffer are never read.
  for (std::int32_t k : st->view.active_batches) {
    auto d_ridx = st->partitioners.At(k)->GetRows();
    dh::LaunchN(d_ridx.size(), ctx_->CUDACtx()->Stream(),
                [=] XGBOOST_DEVICE(std::size_t i) mutable {
                  auto ridx = d_ridx[i];
                  bst_node_t nidx = SamplePosition::Decode(d_position[ridx]);
                  out_preds(ridx, 0) += d_nodes[nidx].LeafValue();
                });
  }
}

void GPUFusedCVHistMaker::PredictValidationBinned(
    DMatrix* p_fmat, std::vector<RegTree const*> const& new_trees,
    std::vector<HostDeviceVector<float>*> const& valid_margins) {
  curt::SetDevice(ctx_->Ordinal());
  std::int32_t const K = this->NumFolds();
  CHECK_EQ(static_cast<std::int32_t>(new_trees.size()), K);
  CHECK_EQ(static_cast<std::int32_t>(valid_margins.size()), K);
  auto stream = ctx_->CUDACtx()->Stream();

  // Concatenate every fold's tree nodes into one device buffer, recording each fold's start
  // offset so the kernel can index the right tree by fold.
  std::vector<RegTree::Node> h_nodes;
  std::vector<std::uint32_t> h_offset(K + 1, 0);
  for (std::int32_t f = 0; f < K; ++f) {
    CHECK(new_trees[f]);
    auto nodes = new_trees[f]->GetNodes(DeviceOrd::CPU());
    h_offset[f + 1] = h_offset[f] + static_cast<std::uint32_t>(nodes.size());
    h_nodes.insert(h_nodes.end(), nodes.begin(), nodes.end());
  }
  dh::DeviceUVector<RegTree::Node> d_nodes;
  dh::CopyTo(h_nodes, &d_nodes, stream);
  dh::DeviceUVector<std::uint32_t> d_offset;
  dh::CopyTo(h_offset, &d_offset, stream);
  dh::DeviceUVector<bst_idx_t> d_valid_ptr;
  dh::CopyTo(folds_.valid_ptr, &d_valid_ptr, stream);

  std::vector<float*> h_margin(K);
  for (std::int32_t f = 0; f < K; ++f) {
    valid_margins[f]->SetDevice(ctx_->Device());
    CHECK_EQ(valid_margins[f]->Size(), folds_.n_rows);
    h_margin[f] = valid_margins[f]->DeviceSpan().data();
  }
  dh::DeviceUVector<float*> d_margin;
  dh::CopyTo(h_margin, &d_margin, stream);

  auto sp_nodes = dh::ToSpan(d_nodes);
  auto sp_offset = dh::ToSpan(d_offset);
  auto sp_valid = dh::ToSpan(d_valid_ptr);
  auto sp_margin = dh::ToSpan(d_margin);
  std::int32_t const n_folds = K;

  // One shared pass over the pages: every row belongs to exactly one fold's validation
  // block, so this predicts all folds' newest trees on their own validation rows at once.
  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx_, StaticBatch(true))) {
    page.Impl()->Visit(ctx_, {}, [&](auto&& d_acc) {
      using Acc = std::remove_reference_t<decltype(d_acc)>;
      CVPredictValidOp<Acc> op{d_acc,    d_acc.base_rowid, sp_nodes, sp_offset,
                               sp_valid, sp_margin,        n_folds};
      dh::LaunchN(d_acc.n_rows, stream, op);
    });
  }
}

void PredictTreeBinned(Context const* ctx, DMatrix* p_fmat, RegTree const& tree,
                       common::Span<float> out_margin) {
  curt::SetDevice(ctx->Ordinal());
  auto stream = ctx->CUDACtx()->Stream();
  CHECK_EQ(out_margin.size(), p_fmat->Info().num_row_);
  auto h_nodes = tree.GetNodes(DeviceOrd::CPU());
  dh::DeviceUVector<RegTree::Node> d_nodes;
  dh::CopyTo(std::vector<RegTree::Node>{h_nodes.begin(), h_nodes.end()}, &d_nodes, stream);
  auto sp_nodes = dh::ToSpan(d_nodes);

  for (auto const& page : p_fmat->GetBatches<EllpackPage>(ctx, StaticBatch(true))) {
    page.Impl()->Visit(ctx, {}, [&](auto&& d_acc) {
      using Acc = std::remove_reference_t<decltype(d_acc)>;
      PredictTreeBinnedOp<Acc> op{d_acc, d_acc.base_rowid, sp_nodes, out_margin};
      dh::LaunchN(d_acc.n_rows, stream, op);
    });
  }
}
}  // namespace xgboost::tree
