/*!
 * Copyright 2019 by Contributors
 * \file multitargets_obj.cu
 */
#include <xgboost/objective.h>

#include "xgboost/span.h"
#include "xgboost/host_device_vector.h"

namespace xgboost {
namespace obj {
class MultiRegLossObj : public ObjFunction {
 public:
  void LoadConfig(Json const& in) override {}
  void SaveConfig(Json* p_out) const override {}
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {}
  void GetGradient(HostDeviceVector<bst_float> const& preds,
                   MetaInfo const& info,
                   int32_t iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    auto const& labels = info.multi_labels_.values_;
    auto const& predts = preds.HostVector();
    CHECK_EQ(labels.size(), preds.Size());
    size_t const n_cols = info.multi_labels_.n_cols;
    size_t const n_rows = info.multi_labels_.n_rows;
    // bool const is_null_weight = info.weights_.Size() == 0;
    out_gpair->Resize(n_rows);

    for (size_t i = 0; i < labels.size(); i += n_cols) {
      for (size_t j = i; j < n_cols; ++j) {
        auto predt = predts[i * n_cols + j];
        auto label = labels[i * n_cols + j];
        out_gpair->HostVector()[i] += GradientPair(predt - label, 1.0);
      }
    }
  }

  void PredTransform(HostDeviceVector<float> *io_preds) override {
    // do nothing
  }

  const char* DefaultEvalMetric() const override {
    return "rmse";
  }
  float ProbToMargin(float base_score) const override {
    return base_score;
  }
};

XGBOOST_REGISTER_OBJECTIVE(SquaredLossRegression, "multi:reg:squarederror")
.describe("Regression with squared error.")
.set_body([]() { return new MultiRegLossObj(); });
}  // namespace obj
}  // namespace xgboost
