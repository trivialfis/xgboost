/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <vector>  // for vector

#include "../helpers.h"  // for MakeCUDACtx
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"

namespace xgboost {
TEST(Tree, ExpandBatch) {
  for (auto ctx : {Context{}, MakeCUDACtx(0)}) {
    RegTree expected, actual;
    HostDeviceVector<float> weights{{1.0f, -2.0f, 3.0f}, ctx.Device()};
    HostDeviceVector<tree::CatWordT> categories{{5}, ctx.Device()};
    auto w = ctx.IsCPU() ? weights.ConstHostSpan() : weights.ConstDeviceSpan();
    auto cats = ctx.IsCPU() ? categories.ConstHostSpan() : categories.ConstDeviceSpan();
    tree::CatWordT host_cats[]{5};

    // Grow the device statistics and category storage across successive batches.
    for (auto nodes : {std::vector<bst_node_t>{0}, std::vector<bst_node_t>{1, 2}}) {
      tree::ExpandBatch batch;
      for (auto nidx : nodes) {
        auto is_cat = nidx != 1;
        tree::SplitInfo split{nidx, 0, 0.5f, nidx != 2, is_cat ? cats : decltype(cats){}};
        batch.push_back(
            {split, {w.subspan(0, 1), 10.0}, {w.subspan(1, 1), 4.0}, {w.subspan(2, 1), 6.0}, 2.0f});
        split.categories = is_cat ? common::Span<tree::CatWordT const>{host_cats}
                                  : common::Span<tree::CatWordT const>{};
        expected.Expand({split, {1.0f, 10.0}, {-2.0f, 4.0}, {3.0f, 6.0}, 2.0f});
      }
      actual.Expand(&ctx, batch);
    }
    expected.FinalizeLeaves(0.3f);
    actual.FinalizeLeaves(0.3f);
    EXPECT_TRUE(actual.Equal(expected));
    for (bst_node_t nidx = 0; nidx < expected.NumNodes(); ++nidx) {
      EXPECT_EQ(actual.Stat(nidx), expected.Stat(nidx));
    }
  }
}
}  // namespace xgboost
