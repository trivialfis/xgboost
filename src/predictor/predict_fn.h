#include "xgboost/tree_model.h"
#include "xgboost/tree_model.h"
#include "../common/categorical.h"

namespace xgboost {
namespace predictor {
inline XGBOOST_DEVICE bst_node_t GetNextNode(
    common::Span<RegTree::Node const> tree, bst_node_t nid, float fvalue,
    bool is_missing, common::Span<FeatureType const> split_types,
    common::Span<uint32_t const> categories,
    common::Span<RegTree::Segment const> cat_ptrs) {
  if (is_missing) {
    nid = tree[nid].DefaultChild();
  } else {
    bool go_left = true;
    if (common::IsCat(split_types, nid)) {
      auto node_categories =
          categories.subspan(cat_ptrs[nid].beg, cat_ptrs[nid].size);
      go_left = Decision(node_categories, common::AsCat(fvalue));
    } else {
      go_left = fvalue < tree[nid].SplitCond();
    }
    if (go_left) {
      nid = tree[nid].LeftChild();
    } else {
      nid = tree[nid].RightChild();
    }
  }
  return nid;
}
}  // namespace predictor
}  // namespace xgboost
