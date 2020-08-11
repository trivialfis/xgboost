/*!
 * Copyright 2019 by XGBoost Contributors
 */
#include <xgboost/parameter.h>
namespace xgboost {
struct PseudoHuberParam : public XGBoostParameter<PseudoHuberParam> {
  float huber_residuals {1.0f};
  DMLC_DECLARE_PARAMETER(PseudoHuberParam) {
    DMLC_DECLARE_FIELD(huber_residuals).set_default(1.0f)
        .describe("Residuals (alpha) for Pseudo Huber loss function.");
  }
};
};  // namespace xgboost
