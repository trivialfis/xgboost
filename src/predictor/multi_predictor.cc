#include "xgboost/predictor.h"
#include "xgboost/base.h"
#include "xgboost/tree_model.h"

#include "../gbm/gbtree_model.h"

namespace xgboost {
namespace predictor {

class NotImplementedError : public dmlc::Error {
 public:
  NotImplementedError(std::string str) : dmlc::Error{str} {};
  char const* what() const noexcept override {
    return "Method is not implemented";
  }
};

class CPUMultiPredictor : public Predictor {
  void PredictLeafValue(common::Span<Entry const> row,
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
        if (split_value < row[split_ind].fvalue) {
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

 public:
  CPUMultiPredictor(GenericParameter const* generic_param,
                    std::shared_ptr<std::unordered_map<DMatrix*, PredictionCacheEntry>> cache)
      : Predictor::Predictor{generic_param, cache} {}
  void PredictBatch(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    auto& h_predt = out_preds->HostVector();
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
    throw NotImplementedError(__func__);
  }
};

XGBOOST_REGISTER_PREDICTOR(CPUMultiPredictor, "multi_predictor")
.describe("Make predictions using CPU.")
.set_body([](GenericParameter const* generic_param,
             std::shared_ptr<std::unordered_map<DMatrix*, PredictionCacheEntry>> cache) {
            return new CPUMultiPredictor(generic_param, cache);
          });

}  // namespace predictor
}  // namespace xgboost
