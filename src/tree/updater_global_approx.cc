#include "xgboost/tree_updater.h"
#include "updater_quantile_hist.h"

namespace xgboost {
namespace tree {
class GlobalApprox : public QuantileHistMaker {
  std::unique_ptr<Builder<float>> builder_;
  // std::unique_ptr<TreeUpdater> pruner_;
  // std::unique_ptr<SplitEvaluator> spliteval_;
  // FeatureInteractionConstraintHost int_constraint_;
  // TrainParam param_;

 public:
  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *dmat,
              const std::vector<RegTree *> &trees) override {
    if (!builder_) {
      builder_.reset(new Builder<float>(
          param_, std::move(pruner_),
          std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone()),
          int_constraint_, dmat));
      if (rabit::IsDistributed()) {
        (builder_)->SetHistSynchronizer(
            new DistributedHistSynchronizer<float>());
        (builder_)->SetHistRowsAdder(new DistributedHistRowsAdder<float>());
      } else {
        (builder_)->SetHistSynchronizer(new BatchHistSynchronizer<float>());
        (builder_)->SetHistRowsAdder(new BatchHistRowsAdder<float>());
      }
    }
    GHistIndexMatrix gradient_index;
    std::vector<float> hessians(gpair->Size());
    auto const &h_gpair = gpair->ConstHostVector();
    std::transform(h_gpair.cbegin(), h_gpair.cend(), hessians.begin(),
                   [](GradientPair const &g) { return g.GetHess(); });
    gradient_index.Init(dmat, hessians, param_.max_bin);
    ColumnMatrix column_matrix;
    column_matrix.Init(gradient_index, param_.sparse_threshold);
    GHistIndexBlockMatrix gmatb;
    if (param_.enable_feature_grouping > 0) {
      gmatb.Init(gmat_, column_matrix_, param_);
    }

    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    int_constraint_.Configure(param_, dmat->Info().num_col_);

    for (auto tree : trees) {
      builder_->Update(gradient_index, gmatb, column_matrix, gpair, dmat, tree);
    }

    param_.learning_rate = lr;
    p_last_dmat_ = dmat;
  }
};

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_global_approx")
.describe("Grow tree using quantized histogram.")
.set_body(
    []() {
      return new GlobalApprox();
    });
}  // namespace tree
}  // namespace xgboost
