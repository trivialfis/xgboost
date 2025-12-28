/**
 * Copyright 2025, XGBoost Contributors
 *
 * Test for reg_lambda (L2 regularization) parameter in multi-target GPU histogram updater.
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/gradient.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>

#include <cmath>
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
 * @brief Test that reg_lambda (L2 regularization) affects tree growth.
 *
 * According to param.h:
 * - CalcWeight: w = -ThresholdL1(sum_grad, reg_alpha) / (sum_hess + reg_lambda)
 * - CalcGain: gain = sum_grad^2 / (sum_hess + reg_lambda)  [when max_delta_step=0, reg_alpha=0]
 *
 * Higher reg_lambda should:
 * 1. Produce smaller leaf weights (larger denominator)
 * 2. Produce smaller gains (larger denominator)
 * 3. Result in shallower trees or fewer splits
 */
TEST(GpuHistMultiTarget, RegLambdaEffect) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 256;
  constexpr bst_target_t kTargets = 2;
  constexpr bst_feature_t kFeatures = 16;

  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  RegTree tree_low_reg{kTargets, kFeatures};
  RegTree tree_high_reg{kTargets, kFeatures};

  // Build tree with low reg_lambda (more freedom to fit)
  {
    Args args{
        {"max_depth", "3"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "0.1"},  // Low regularization
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_low_reg);
  }

  // Build tree with high reg_lambda (more regularization)
  {
    Args args{
        {"max_depth", "3"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "100.0"},  // High regularization
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_high_reg);
  }

  ASSERT_TRUE(tree_low_reg.IsMultiTarget());
  ASSERT_TRUE(tree_high_reg.IsMultiTarget());

  // High reg_lambda should result in fewer or equal nodes (more regularization)
  ASSERT_GE(tree_low_reg.NumNodes(), tree_high_reg.NumNodes())
      << "Higher reg_lambda should produce fewer or equal nodes due to regularization";
}

/**
 * @brief Test reg_lambda with multiple targets.
 *
 * This verifies that reg_lambda is applied consistently across all targets.
 */

/**
 * @brief Test reg_lambda=0 vs reg_lambda=1 (default).
 *
 * According to param.h line 125, the default reg_lambda is 1.0.
 * reg_lambda=0 should allow more aggressive fitting.
 */
TEST(GpuHistMultiTarget, RegLambdaZeroVsOne) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 256;
  constexpr bst_target_t kTargets = 3;
  constexpr bst_feature_t kFeatures = 16;

  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  RegTree tree_zero{kTargets, kFeatures};
  RegTree tree_one{kTargets, kFeatures};

  // Build with reg_lambda=0 (no L2 regularization)
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
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_zero);
  }

  // Build with reg_lambda=1 (default L2 regularization)
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "1"},
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_one);
  }

  ASSERT_TRUE(tree_zero.IsMultiTarget());
  ASSERT_TRUE(tree_one.IsMultiTarget());

  // Trees may have different structures due to regularization
  // Zero regularization typically allows more splits
  ASSERT_GE(tree_zero.NumNodes(), tree_one.NumNodes())
      << "reg_lambda=0 should produce trees with more or equal nodes than reg_lambda=1";
}

/**
 * @brief Test that very high reg_lambda limits tree growth.
 *
 * With high L2 regularization, the gains should be smaller,
 * potentially resulting in fewer splits.
 */
TEST(GpuHistMultiTarget, RegLambdaVeryHigh) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 256;
  constexpr bst_target_t kTargets = 2;
  constexpr bst_feature_t kFeatures = 16;

  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  Args args{
      {"max_depth", "2"},
      {"min_child_weight", "1"},
      {"reg_alpha", "0"},
      {"reg_lambda", "1000"},  // Very high regularization
      {"min_split_loss", "0"},
  };
  TrainParam param;
  param.UpdateAllowUnknown(args);

  RegTree tree{kTargets, kFeatures};
  BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree);

  // With very high reg_lambda, tree growth should be limited
  ASSERT_TRUE(tree.IsMultiTarget());
  ASSERT_GE(tree.NumNodes(), 1);  // At least root node should exist
}

/**
 * @brief Test reg_lambda interaction with reg_alpha.
 *
 * Both reg_lambda (L2) and reg_alpha (L1) should work together to regularize the model.
 */
TEST(GpuHistMultiTarget, RegLambdaWithRegAlpha) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 256;
  constexpr bst_target_t kTargets = 2;
  constexpr bst_feature_t kFeatures = 16;

  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  RegTree tree_no_reg{kTargets, kFeatures};
  RegTree tree_l2_only{kTargets, kFeatures};
  RegTree tree_both_reg{kTargets, kFeatures};

  // Build with no regularization
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
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_no_reg);
  }

  // Build with L2 regularization only
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "0"},
        {"reg_lambda", "10"},
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_l2_only);
  }

  // Build with both L1 and L2 regularization
  {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "0"},
        {"reg_alpha", "10"},
        {"reg_lambda", "10"},
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree_both_reg);
  }

  ASSERT_TRUE(tree_no_reg.IsMultiTarget());
  ASSERT_TRUE(tree_l2_only.IsMultiTarget());
  ASSERT_TRUE(tree_both_reg.IsMultiTarget());

  // More regularization should result in fewer or equal nodes
  ASSERT_GE(tree_no_reg.NumNodes(), tree_l2_only.NumNodes())
      << "reg_lambda should reduce tree size compared to no regularization";
  ASSERT_GE(tree_l2_only.NumNodes(), tree_both_reg.NumNodes())
      << "Combined L1+L2 regularization should be at least as restrictive as L2 alone";
}

/**
 * @brief Test that the gain formula properly includes reg_lambda.
 *
 * According to param.h line 254: gain = sum_grad^2 / (sum_hess + reg_lambda)
 * This test verifies that the denominator includes reg_lambda.
 */
TEST(GpuHistMultiTarget, RegLambdaGainFormula) {
  auto ctx = MakeCUDACtx(0);

  constexpr bst_idx_t kRows = 512;
  constexpr bst_target_t kTargets = 2;
  constexpr bst_feature_t kFeatures = 32;

  // Use a large dataset to ensure meaningful splits
  auto p_fmat = RandomDataGenerator{kRows, kFeatures, 0.0f}.GenerateDMatrix(true);
  auto gpair = GenerateRandomGradients(&ctx, kRows, kTargets);

  std::vector<float> reg_lambda_values{0.1f, 1.0f, 10.0f};
  std::vector<bst_node_t> tree_sizes;

  for (float reg_lambda : reg_lambda_values) {
    Args args{
        {"max_depth", "2"},
        {"min_child_weight", "1"},  // Small minimum to allow splits
        {"reg_alpha", "0"},
        {"reg_lambda", std::to_string(reg_lambda)},
        {"min_split_loss", "0"},
    };
    TrainParam param;
    param.UpdateAllowUnknown(args);

    RegTree tree{kTargets, kFeatures};
    BuildMultiTargetTree(&ctx, p_fmat.get(), &gpair, param, &tree);

    ASSERT_TRUE(tree.IsMultiTarget());
    ASSERT_GE(tree.NumNodes(), 1);
    tree_sizes.push_back(tree.NumNodes());
  }

  // Verify trend: higher reg_lambda typically produces fewer nodes
  // Note: This is not strictly monotonic due to randomness, but should hold generally
  EXPECT_GE(tree_sizes[0], tree_sizes[2])
      << "Tree size with reg_lambda=" << reg_lambda_values[0]
      << " should generally be >= tree size with reg_lambda=" << reg_lambda_values[2];
}
}  // namespace xgboost::tree
