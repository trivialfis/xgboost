#include "xgboost/predictor.h"
#include "xgboost/base.h"
#include "xgboost/tree_model.h"

#include "../gbm/gbtree_model.h"

namespace xgboost {
namespace predictor {

class CPUMultiPredictor : public Predictor {
  void PredictLeafValue(common::Span<float const> row,
                        const std::vector<std::unique_ptr<RegTree>>& trees,
                        const std::vector<int>& tree_info,
                        unsigned tree_begin, unsigned tree_end,
                        common::Span<float> prediction) {
    for (size_t i = tree_begin; i < tree_end; ++i) {
      auto node_id = 0;
      auto& tree = *trees[i];
      while(!tree[node_id].IsLeaf()) {
        float split_value = tree[node_id].SplitCond();
        bst_feature_t split_ind = tree[node_id].SplitIndex();
        if (split_value < row[split_ind]) {
          node_id = tree[node_id].LeftChild();
        } else {
          node_id = tree[node_id].RightChild();
        }
      }

      for (size_t i = 0; i < prediction.size(); ++i) {
        prediction[i] += tree.LeafValueVector(node_id)[i];
      }
    }
  }

  void PredictBatch(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    auto const& h_predt = out_preds->ConstHostVector();
    common::Span<float> predictions {h_predt};
    for (const auto& batch : dmat->GetBatches<SparsePage>()) {
      for (size_t i = 0; i < batch.Size(); ++i) {
        this->PredictLeafValue(batch[i],
                               model.trees, model.tree_info,
                               tree_begin, ntree_limit, predictions);
      }
    }
  }

  void UpdatePredictionCache(
      const gbm::GBTreeModel& model,
      std::vector<std::unique_ptr<TreeUpdater>>* updaters,
      int num_new_trees) override {

  }

  void PredictInstance(const SparsePage::Inst &inst,
                       std::vector<bst_float> *out_preds,
                       const gbm::GBTreeModel &model,
                       unsigned ntree_limit = 0) override {

  }

  void PredictLeaf(DMatrix *dmat, std::vector<bst_float> *out_preds,
                   const gbm::GBTreeModel &model,
                   unsigned ntree_limit = 0) override {}

  void PredictContribution(DMatrix *dmat, std::vector<bst_float> *out_contribs,
                           const gbm::GBTreeModel &model, unsigned ntree_limit,
                           std::vector<bst_float> *tree_weights,
                           bool approximatek, int condition,
                           unsigned condition_feature) override {

  }

  void PredictInteractionContributions(DMatrix *dmat,
                                       std::vector<bst_float> *out_contribs,
                                       const gbm::GBTreeModel &model,
                                       unsigned ntree_limit,
                                       std::vector<bst_float> *tree_weights,
                                       bool approximate) override {

  }
};

}
}
