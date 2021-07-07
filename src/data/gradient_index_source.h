#ifndef XGBOOST_DATA_GRADIENT_INDEX_SOURCE_H_
#define XGBOOST_DATA_GRADIENT_INDEX_SOURCE_H_

#include "sparse_page_source.h"

namespace xgboost {
namespace data {
class GHistIndexSource : public PageSourceIncMixIn<GHistIndexMatrix> {

 public:
  GHistIndexSource(float missing, int nthreads, bst_feature_t n_features,
                   size_t n_batches, std::shared_ptr<Cache> cache)
      : PageSourceIncMixIn(missing, nthreads, n_features, n_batches, cache) {}
};
}      // namespace data
}      // namespace xgboost
#endif  // XGBOOST_DATA_GRADIENT_INDEX_SOURCE_H_
