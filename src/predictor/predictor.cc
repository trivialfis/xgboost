/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/registry.h>
#include <xgboost/predictor.h>

namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::PredictorReg);
}  // namespace dmlc
namespace xgboost {
void Predictor::Configure(
    const std::vector<std::pair<std::string, std::string>>& cfg) {}
Predictor* Predictor::Create(std::string const& name,
                             std::unordered_map<DMatrix*, PredictionCacheEntry>* _cache,
                             GenericParameter const* learner_param) {
  auto* e = ::dmlc::Registry<PredictorReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown predictor type " << name;
  }
  auto p_predictor =  (e->body)();
  p_predictor->learner_param_ = learner_param;
  p_predictor->SetCache(_cache);
  return p_predictor;
}
}  // namespace xgboost

namespace xgboost {
namespace predictor {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(gpu_predictor);
#endif  // XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(cpu_predictor);
}  // namespace predictor
}  // namespace xgboost
