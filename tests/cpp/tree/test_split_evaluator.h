/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <gtest/gtest.h>

#include <cstdint>  // for int32_t
#include <limits>   // for numeric_limits
#include <vector>   // for vector

#include "../../../src/common/cuda_stream.h"
#include "../../../src/tree/split_evaluator.h"

namespace xgboost::tree {
template <bool CompiledWithCuda = WITH_CUDA()>
void TestBatchedSplitBounds(DeviceOrd device) {
  constexpr float kLimit = std::numeric_limits<float>::max();
  for (bst_target_t n_targets : {1, 3}) {
    SCOPED_TRACE(n_targets);
    TrainParam param;
    param.Init(Args{{"monotone_constraints", "(1, -1, 0)"}});
    TreeEvaluator tree_evaluator{param, 3, device, n_targets};

    HostDeviceVector<float> left(n_targets, 0.0f, device);
    HostDeviceVector<float> right(n_targets, 0.0f, device);
    for (bst_target_t t = 0; t < n_targets; ++t) {
      left.HostVector()[t] = 10.0f * t - 4.0f;
      right.HostVector()[t] = 10.0f * t + 4.0f;
    }
    if (n_targets == 1) {
      tree_evaluator.AddSplit(0, 1, 2, 0, -4.0f, 4.0f);
    } else {
      auto l = device.IsCPU() ? left.ConstHostSpan() : left.ConstDeviceSpan();
      auto r = device.IsCPU() ? right.ConstHostSpan() : right.ConstDeviceSpan();
      tree_evaluator.AddSplit(0, 1, 2, 0, linalg::MakeVec(device, l), linalg::MakeVec(device, r));
    }

    auto apply_batch = [&](std::vector<std::int32_t> nodes, float first_mid, float second_mid,
                           bst_node_t max_nidx) {
      // Each record contains parent, left child, right child, and feature index.
      HostDeviceVector<std::int32_t> node_storage{nodes, device};
      HostDeviceVector<float> mid_storage(2 * n_targets, 0.0f, device);
      for (bst_target_t t = 0; t < n_targets; ++t) {
        mid_storage.HostVector()[t] = 10.0f * t + first_mid;
        mid_storage.HostVector()[n_targets + t] = 10.0f * t + second_mid;
      }
      auto splits = device.IsCPU() ? node_storage.ConstHostSpan() : node_storage.ConstDeviceSpan();
      auto mids = device.IsCPU() ? mid_storage.ConstHostSpan() : mid_storage.ConstDeviceSpan();
      auto get_split = [=] XGBOOST_DEVICE(std::size_t i, bst_target_t t) {
        auto offset = 4 * i;
        return TreeEvaluator::SplitBounds{splits[offset], splits[offset + 1], splits[offset + 2],
                                          static_cast<bst_feature_t>(splits[offset + 3]),
                                          mids[i * n_targets + t]};
      };
      tree_evaluator.AddSplits(2, max_nidx, get_split);
      tree_evaluator.AddSplits(0, max_nidx, get_split);
      // Keep input storage alive until the device has consumed it.
#if defined(__CUDACC__)
      curt::DefaultStream().Sync();
#endif
    };

    // Decreasing and unconstrained splits must both inherit the root's increasing bounds.
    apply_batch({1, 3, 4, 1, 2, 5, 6, 2}, -2.0f, 2.0f, 6);
    // Nonconsecutive parents and large child IDs exercise bound-buffer growth.
    apply_batch({3, 300, 301, 2, 5, 302, 303, 1}, -1.0f, 3.0f, 303);

    auto evaluator = tree_evaluator.GetEvaluator();
    HostDeviceVector<float> bounds(8 * n_targets, 0.0f, device);
    common::Transform<CompiledWithCuda>::Init(
        [=] XGBOOST_DEVICE(std::size_t i, common::Span<float> out) {
          auto nidx = static_cast<bst_node_t>(300 + i / n_targets);
          auto t = static_cast<bst_target_t>(i % n_targets);
          out[2 * i] = evaluator.ApplyBounds(nidx, t, -kLimit);
          out[2 * i + 1] = evaluator.ApplyBounds(nidx, t, kLimit);
        },
        common::Range(0, 4 * n_targets), 1, device)
        .Eval(&bounds);
    auto result = bounds.ConstHostSpan();
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto offset = 10.0f * t;
      float expected[4][2]{{offset - 2.0f, offset},
                           {offset - 2.0f, offset},
                           {offset + 3.0f, kLimit},
                           {offset, offset + 3.0f}};
      for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(result[2 * (i * n_targets + t)], expected[i][0]);
        EXPECT_EQ(result[2 * (i * n_targets + t) + 1], expected[i][1]);
      }
    }
  }

  TrainParam param;
  param.Init(Args{});
  TreeEvaluator unconstrained{param, 1, device, 1};
  EXPECT_FALSE(unconstrained.HasConstraint());
  // An unconstrained evaluator must not invoke the accessor or allocate child bounds.
  unconstrained.AddSplits(2, 303, [] XGBOOST_DEVICE(std::size_t, bst_target_t) {
    return TreeEvaluator::SplitBounds{-1, -1, -1, 0, 0.0f};
  });
  EXPECT_FALSE(unconstrained.GetEvaluator().has_constraint);
}
}  // namespace xgboost::tree
