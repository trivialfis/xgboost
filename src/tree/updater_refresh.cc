/**
 * Copyright 2014-2025, XGBoost Contributors
 * \file updater_refresh.cc
 * \brief refresh the statistics and leaf value on the tree on the dataset
 * \author Tianqi Chen
 */
#include <algorithm>  // for fill
#include <cstdint>    // for int32_t
#include <vector>     // for vector

#include "../collective/allreduce.h"     // for Allreduce
#include "../common/threading_utils.h"   // for ParallelFor
#include "../predictor/predict_fn.h"     // for GetNextNode
#include "./param.h"                     // for TrainParam
#include "xgboost/data.h"                // for DMatrix, SparsePage
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json
#include "xgboost/linalg.h"              // for Matrix
#include "xgboost/span.h"                // for Span
#include "xgboost/tree_updater.h"        // for TreeUpdater

namespace xgboost::tree {

DMLC_REGISTRY_FILE_TAG(updater_refresh);

/*! \brief pruner that prunes a tree after growing finishes */
class TreeRefresher : public TreeUpdater {
 public:
  explicit TreeRefresher(Context const *ctx) : TreeUpdater(ctx) {}
  void Configure(const Args &) override {}
  void LoadConfig(Json const &) override {}
  void SaveConfig(Json *) const override {}

  [[nodiscard]] char const *Name() const override { return "refresh"; }
  [[nodiscard]] bool CanModifyTree() const override { return true; }
  // Update the tree.
  void Update(TrainParam const *param, linalg::Matrix<GradientPair> *gpair, DMatrix *p_fmat,
              common::Span<HostDeviceVector<bst_node_t>> /*out_position*/,
              const std::vector<RegTree *> &trees) override {
    if (trees.empty()) {
      return;
    }
    CHECK_EQ(gpair->Shape(1), 1) << MTNotImplemented();
    auto const &gpair_h = gpair->Data()->ConstHostVector();

    // Setup temp space for each thread
    const std::int32_t n_threads = ctx_->Threads();
    std::vector<std::vector<GradStats>> stemp(n_threads);
    std::vector<RegTree::FVec> fvec_temp(n_threads);

    bst_node_t n_nodes = 0;
    for (auto tree : trees) {
      auto n = tree->NumNodes();
      CHECK_GE(n_nodes + n, n_nodes);
      n_nodes += n;
    }

    dmlc::OMPException exc;
#pragma omp parallel num_threads(n_threads)
    {
      exc.Run([&]() {
        auto tid = omp_get_thread_num();
        stemp[tid].resize(n_nodes);
        std::fill(stemp[tid].begin(), stemp[tid].end(), GradStats{});
        fvec_temp[tid].Init(trees[0]->NumFeatures());
      });
    }
    exc.Rethrow();

    auto get_stats = [&]() {
      // start accumulating statistics
      for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
        auto page = batch.GetView();
        common::ParallelFor(batch.Size(), ctx_->Threads(), [&](auto i) {
          SparsePage::Inst inst = page[i];
          const int tid = omp_get_thread_num();
          const auto ridx = batch.base_rowid + i;
          RegTree::FVec &feats = fvec_temp[tid];
          feats.Fill(inst);
          int offset = 0;
          for (auto tree : trees) {
            AddStats(*tree, feats, gpair_h, ridx, dmlc::BeginPtr(stemp[tid]) + offset);
            offset += tree->NumNodes();
          }
          feats.Drop();
        });
      }
      // aggregate the statistics
      auto num_nodes = static_cast<int>(stemp[0].size());
      common::ParallelFor(num_nodes, ctx_->Threads(), [&](int nid) {
        for (int tid = 1; tid < n_threads; ++tid) {
          stemp[0][nid].Add(stemp[tid][nid]);
        }
      });
    };
    get_stats();
    // Synchronize the aggregated result.
    auto &sum_grad = stemp[0];
    // x2 for gradient and hessian.
    auto rc = collective::Allreduce(
        ctx_, linalg::MakeVec(&sum_grad.data()->sum_grad, sum_grad.size() * 2),
        collective::Op::kMax);
    collective::SafeColl(rc);
    bst_node_t offset = 0;
    for (auto tree : trees) {
      this->Refresh(param, dmlc::BeginPtr(sum_grad) + offset, 0, tree);
      offset += tree->NumNodes();
    }
  }

 private:
  static void AddStats(RegTree const &tree, RegTree::FVec const &feat,
                       std::vector<GradientPair> const &gpair, bst_idx_t ridx, GradStats *gstats) {
    // start from groups that belongs to current data
    auto pid = 0;
    gstats[pid].Add(gpair[ridx]);
    auto const &cats = tree.GetCategoriesMatrix();
    // traverse tree
    while (!tree[pid].IsLeaf()) {
      auto split_index = tree[pid].SplitIndex();
      pid = predictor::GetNextNode<true, true>(tree[pid], pid, feat.GetFvalue(split_index),
                                               feat.IsMissing(split_index), cats);
      gstats[pid].Add(gpair[ridx]);
    }
  }

  void Refresh(TrainParam const *param, const GradStats *gstats, int nid, RegTree *p_tree) const {
    RegTree &tree = *p_tree;
    tree.Stat(nid).base_weight = static_cast<float>(CalcWeight(*param, gstats[nid]));
    tree.Stat(nid).sum_hess = static_cast<float>(gstats[nid].sum_hess);
    if (tree[nid].IsLeaf()) {
      if (param->refresh_leaf) {
        tree[nid].SetLeaf(tree.Stat(nid).base_weight * param->learning_rate);
      }
      return;
    }

    tree.Stat(nid).loss_chg =
        static_cast<float>(xgboost::tree::CalcGain(*param, gstats[tree[nid].LeftChild()]) +
                           xgboost::tree::CalcGain(*param, gstats[tree[nid].RightChild()]) -
                           xgboost::tree::CalcGain(*param, gstats[nid]));
    this->Refresh(param, gstats, tree[nid].LeftChild(), p_tree);
    this->Refresh(param, gstats, tree[nid].RightChild(), p_tree);
  }
};

XGBOOST_REGISTER_TREE_UPDATER(TreeRefresher, "refresh")
    .describe("Refresher that refreshes the weight and statistics according to data.")
    .set_body([](Context const *ctx, auto) { return new TreeRefresher(ctx); });
}  // namespace xgboost::tree
