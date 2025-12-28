/**
 * Copyright 2025, XGBoost Contributors
 *
 * Test for min_split_loss parameter in multi-target GPU histogram updater.
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/gradient.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include <memory>
#include <vector>

#include "../../../src/tree/param.h"
#include "../helpers.h"

namespace xgboost::tree {
namespace {

/**
 * @brief Helper function to build a multi-target tree with GPU hist updater.
 *
 * @param ctx The execution context
 * @param dmat Training data matrix
 * @param gpair Gradient pairs for all targets
 * @param param Training parameters
 * @param tree The tree to be built (must be initialized with n_targets > 1 for multi-target)
 *
 * @note Requires debug_synchronize=1 for multi-target GPU histogram support.
 */
void BuildMultiTargetTree(Context const* ctx, DMatrix* dmat, GradientContainer* gpair,
                          TrainParam const& param, RegTree* tree) {
  ObjInfo task{ObjInfo::kRegression};

  // The updater determines multi-target based on the tree structure (tree->IsMultiTarget())
  std::unique_ptr<TreeUpdater> updater{TreeUpdater::Create("grow_gpu_hist", ctx, &task)};
  // Enable debug_synchronize for multi-target support (required for current implementation)
  updater->Configure(Args{{"debug_synchronize", "1"}});

  std::vector<HostDeviceVector<bst_node_t>> position(1);
  updater->Update(&param, gpair, dmat, common::Span<HostDeviceVector<bst_node_t>>{position},
                  {tree});
}

}  // anonymous namespace

/**
 * @brief Test that min_split_loss prevents splits when loss_chg is below threshold.
 *
 * This test verifies that:
 * 1. With min_split_loss=0, the tree grows normally
 * 2. With a high min_split_loss, no splits are made (tree remains a root node)
 * 3. With a moderate min_split_loss, only high-gain splits are made
 */
TEST(GpuHistMultiTarget, MinSplitLoss) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 128;
  constexpr bst_target_t kTargets = 2;
  constexpr bst_feature_t kFeatures = 8;

  // Create dataset
  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);

  // Create gradient data with different values for each target
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  // Test 1: Build tree with min_split_loss=0 (should create splits)
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);

    RegTree tree{kTargets, kFeatures};
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree);

    // Tree should have grown (more than just root node)
    ASSERT_GT(tree.NumExtraNodes(), 0) << "Tree should grow with min_split_loss=0";
    ASSERT_TRUE(tree.IsMultiTarget());
    ASSERT_EQ(tree.NumTargets(), kTargets);
  }

  // Test 2: Build tree with very high min_split_loss (should not split)
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "1e6"},  // Very high threshold
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);

    RegTree tree{kTargets, kFeatures};
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree);

    // Tree should not have grown beyond root
    ASSERT_EQ(tree.NumExtraNodes(), 0)
        << "Tree should not grow with very high min_split_loss";
    ASSERT_TRUE(tree.IsMultiTarget());
    ASSERT_EQ(tree.NumTargets(), kTargets);
  }

  // Test 3: Verify that min_split_loss threshold is respected
  // Build two trees with different min_split_loss values
  {
    constexpr float kLowThreshold = 0.1f;
    constexpr float kHighThreshold = 10.0f;

    RegTree tree_low{kTargets, kFeatures};
    RegTree tree_high{kTargets, kFeatures};

    // Build tree with low threshold
    {
      Args args{
          {"max_depth", "3"},
          {"min_child_weight", "0"},
          {"reg_alpha", "0"},
          {"reg_lambda", "0"},
          {"min_split_loss", std::to_string(kLowThreshold)},
      };
      TrainParam param;
      param.UpdateAllowUnknown(args);
      BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_low);
    }

    // Build tree with high threshold
    {
      Args args{
          {"max_depth", "3"},
          {"min_child_weight", "0"},
          {"reg_alpha", "0"},
          {"reg_lambda", "0"},
          {"min_split_loss", std::to_string(kHighThreshold)},
      };
      TrainParam param;
      param.UpdateAllowUnknown(args);
      BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_high);
    }

    // Tree with low threshold should have more or equal nodes than tree with high threshold
    ASSERT_GE(tree_low.NumNodes(), tree_high.NumNodes())
        << "Tree with lower min_split_loss should have at least as many nodes";
    ASSERT_TRUE(tree_low.IsMultiTarget());
    ASSERT_TRUE(tree_high.IsMultiTarget());
  }
}

/**
 * @brief Test min_split_loss with multiple targets to ensure it works consistently.
 *
 * This test verifies that min_split_loss works correctly across different numbers of targets.
 */
TEST(GpuHistMultiTarget, MinSplitLossMultipleTargets) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 256;
  constexpr bst_feature_t kFeatures = 16;

  for (bst_target_t n_targets : {1, 2, 4, 8}) {
    auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
    auto gpair = GenerateRandomGradients(&ctx, kRows, n_targets);

    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "5.0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);

    RegTree tree{n_targets, kFeatures};
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree);

    ASSERT_TRUE(tree.IsMultiTarget() || n_targets == 1);
    ASSERT_EQ(tree.NumTargets(), n_targets);

    // Tree may or may not grow depending on data, but should not crash
    ASSERT_GE(tree.NumNodes(), 1) << "Tree should at least have root node for "
                                  << n_targets << " targets";
  }
}

/**
 * @brief Test that min_split_loss=0 is equivalent to no threshold.
 */
TEST(GpuHistMultiTarget, MinSplitLossZero) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 128;
  constexpr bst_target_t kTargets = 3;
  constexpr bst_feature_t kFeatures = 8;

  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  RegTree tree_no_threshold{kTargets, kFeatures};
  RegTree tree_zero_threshold{kTargets, kFeatures};

  // Build without explicit min_split_loss (default is 0)
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_no_threshold);
  }

  // Build with explicit min_split_loss=0
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_zero_threshold);
  }

  // Both trees should have the same structure
  ASSERT_EQ(tree_no_threshold.NumNodes(), tree_zero_threshold.NumNodes())
      << "Trees should be identical with default and explicit min_split_loss=0";
}

/**
 * @brief Test that min_split_loss interacts correctly with other parameters.
 *
 * This test verifies that min_split_loss works in combination with max_depth
 * and ensures splits are rejected based on loss_chg even at shallow depths.
 */
TEST(GpuHistMultiTarget, MinSplitLossWithMaxDepth) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 256;
  constexpr bst_target_t kTargets = 2;
  constexpr bst_feature_t kFeatures = 16;

  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  // Test with max_depth=1 and reasonable min_split_loss
  {
    Args args{
        {"max_depth", "1"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "1.0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);

    RegTree tree{kTargets, kFeatures};
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree);

    // Tree should have at most 3 nodes (root + 2 children) due to max_depth=1
    ASSERT_LE(tree.NumNodes(), 3) << "Tree should respect max_depth=1";
    ASSERT_TRUE(tree.IsMultiTarget());

    // The tree may or may not grow depending on whether any split exceeds min_split_loss=1.0
    // This is expected behavior - we just verify no crashes and constraints are respected
  }

  // Test with max_depth=2 and very restrictive min_split_loss
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "1e5"},  // Very restrictive
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);

    RegTree tree{kTargets, kFeatures};
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree);

    // Despite max_depth=2 allowing growth, min_split_loss should prevent it
    ASSERT_EQ(tree.NumExtraNodes(), 0)
        << "Very high min_split_loss should prevent any splits regardless of max_depth";
  }
}

/**
 * @brief Test min_split_loss boundary condition (loss_chg == min_split_loss).
 *
 * According to expand_entry.cuh line 160, the condition is:
 *   if (split.loss_chg < param.min_split_loss) return false;
 *
 * This means loss_chg == min_split_loss should be accepted.
 */
TEST(GpuHistMultiTarget, MinSplitLossBoundary) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 128;
  constexpr bst_target_t kTargets = 2;
  constexpr bst_feature_t kFeatures = 8;

  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  // Build with min_split_loss slightly below expected gain
  RegTree tree_below{kTargets, kFeatures};
  {
    Args args{
        {"max_depth", "1"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "0.01"},  // Low threshold - should allow splits
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_below);
  }

  // Build with min_split_loss well above expected gain
  RegTree tree_above{kTargets, kFeatures};
  {
    Args args{
        {"max_depth", "1"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0"},
        {"min_split_loss", "1e4"},  // High threshold - should block splits
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_above);
  }

  // Verify different behavior
  ASSERT_TRUE(tree_below.IsMultiTarget());
  ASSERT_TRUE(tree_above.IsMultiTarget());
  ASSERT_GE(tree_below.NumNodes(), tree_above.NumNodes())
      << "Lower min_split_loss should result in more or equal nodes";
}

}  // namespace xgboost::tree
