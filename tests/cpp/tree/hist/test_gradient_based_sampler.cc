/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>  // for size_t
#include <string>   // for to_string

#include "../../../../src/tree/hist/sampler.h"  // for SampleGradient
#include "../../../../src/tree/param.h"         // for TrainParam
#include "../../helpers.h"                      // for GenerateRandomGradients
#include "xgboost/base.h"                       // for GradientPair, bst_target_t
#include "xgboost/context.h"                    // for Context
#include "xgboost/linalg.h"                     // for Matrix, Constant

namespace xgboost::tree {
/**
 * @brief Test gradient-based sampling for CPU.
 */
TEST(CPUGradientBasedSampler, Basic) {
  std::size_t constexpr kRows = 2048;
  double constexpr kSubsample = 0.5;
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", std::to_string(kSubsample)},
                                {"sampling_method", "gradient_based"}});
  Context ctx;

  auto run = [&](bst_target_t n_targets) {
    // Generate random gradient pairs with magnitudes in [0, 1] (same as GPU test)
    auto init_gpairs = GenerateRandomGradients(kRows * n_targets, 0.0f, 1.0f);
    std::size_t shape[2] = {kRows, static_cast<std::size_t>(n_targets)};
    linalg::Matrix<GradientPair> gpair{init_gpairs.HostVector().begin(),
                                       init_gpairs.HostVector().end(), shape, DeviceOrd::CPU()};
    auto h_gpair = gpair.HostView();

    // Calculate original gradient sum
    std::vector<GradientPairPrecise> original_sum(n_targets);
    for (std::size_t i = 0; i < kRows; ++i) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        original_sum[t] += GradientPairPrecise{h_gpair(i, t)};
      }
    }

    // Apply sampling
    SampleGradient(&ctx, param, h_gpair);

    // Calculate sampled gradient sum
    std::vector<GradientPairPrecise> sampled_sum(n_targets);
    std::size_t n_sampled = 0;
    for (std::size_t i = 0; i < kRows; ++i) {
      bool sampled = false;
      if (h_gpair(i, 0).GetGrad() != 0.0f || h_gpair(i, 0).GetHess() != 0.0f) {
        sampled = true;
        n_sampled++;
      }
      // Verify all targets in a row are sampled/zeroed consistently
      for (bst_target_t t = 0; t < n_targets; ++t) {
        bool is_zero = (h_gpair(i, t).GetGrad() == 0.0f && h_gpair(i, t).GetHess() == 0.0f);
        ASSERT_EQ(sampled, !is_zero);
        if (sampled) {
          sampled_sum[t] += GradientPairPrecise{h_gpair(i, t)};
        }
      }
    }

    // Verify approximately the right fraction of rows are sampled
    auto ratio = static_cast<double>(n_sampled) / static_cast<double>(kRows);
    EXPECT_NEAR(ratio, kSubsample, 0.1);

    // Verify gradient sums are approximately preserved (within tolerance)
    // Gradient-based sampling with inverse probability weighting should preserve sums
    // Use same tolerance as GPU tests: 0.03 * kRows per target
    float tolerance = 0.03f * kRows;
    for (bst_target_t t = 0; t < n_targets; ++t) {
      EXPECT_NEAR(original_sum[t].GetGrad(), sampled_sum[t].GetGrad(), tolerance);
      EXPECT_NEAR(original_sum[t].GetHess(), sampled_sum[t].GetHess(), tolerance);
    }
  };

  // Test single target
  run(1);
  // Test multi-target
  run(3);
  run(5);
}

/**
 * @brief Test that gradient-based sampling with subsample=1.0 doesn't modify gradients.
 */
TEST(CPUGradientBasedSampler, NoSampling) {
  std::size_t constexpr kRows = 256;
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"subsample", "1.0"}, {"sampling_method", "gradient_based"}});
  Context ctx;

  auto init = GradientPair{1.0f, 2.0f};
  linalg::Matrix<GradientPair> gpair = linalg::Constant(&ctx, init, kRows, std::size_t{2});
  auto h_gpair = gpair.HostView();

  SampleGradient(&ctx, param, h_gpair);

  // All gradients should remain unchanged
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t t = 0; t < 2; ++t) {
      ASSERT_FLOAT_EQ(h_gpair(i, t).GetGrad(), init.GetGrad());
      ASSERT_FLOAT_EQ(h_gpair(i, t).GetHess(), init.GetHess());
    }
  }
}

/**
 * @brief Test with very low subsample rate.
 */
TEST(CPUGradientBasedSampler, LowSubsampleRate) {
  std::size_t constexpr kRows = 4096;
  double constexpr kSubsample = 0.1;
  TrainParam param;
  param.UpdateAllowUnknown(
      Args{{"subsample", std::to_string(kSubsample)}, {"sampling_method", "gradient_based"}});
  Context ctx;

  auto init_gpairs = GenerateRandomGradients(kRows, -5.0f, 5.0f);
  std::size_t shape[2] = {kRows, std::size_t{1}};
  linalg::Matrix<GradientPair> gpair{init_gpairs.HostVector().begin(),
                                     init_gpairs.HostVector().end(), shape, DeviceOrd::CPU()};
  auto h_gpair = gpair.HostView();

  SampleGradient(&ctx, param, h_gpair);

  std::size_t n_sampled = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    if (h_gpair(i, 0).GetGrad() != 0.0f || h_gpair(i, 0).GetHess() != 0.0f) {
      n_sampled++;
    }
  }

  auto ratio = static_cast<double>(n_sampled) / static_cast<double>(kRows);
  EXPECT_NEAR(ratio, kSubsample, 0.05);
}

/**
 * @brief Test that rows with larger gradients are more likely to be sampled.
 *
 * This verifies the gradient-based nature of the sampling.
 */
TEST(CPUGradientBasedSampler, GradientImportance) {
  std::size_t constexpr kRows = 1024;
  std::size_t constexpr kLargeGradRows = 128;
  double constexpr kSubsample = 0.2;
  TrainParam param;
  param.UpdateAllowUnknown(
      Args{{"subsample", std::to_string(kSubsample)}, {"sampling_method", "gradient_based"}});
  Context ctx;

  // Create gradients where first kLargeGradRows have large gradients
  std::size_t shape[2] = {kRows, std::size_t{1}};
  linalg::Matrix<GradientPair> gpair(shape, DeviceOrd::CPU());
  auto h_gpair = gpair.HostView();

  for (std::size_t i = 0; i < kRows; ++i) {
    if (i < kLargeGradRows) {
      h_gpair(i, 0) = GradientPair{10.0f, 5.0f};  // Large gradient
    } else {
      h_gpair(i, 0) = GradientPair{0.1f, 0.05f};  // Small gradient
    }
  }

  SampleGradient(&ctx, param, h_gpair);

  // Count sampled rows in each group
  std::size_t large_sampled = 0;
  std::size_t small_sampled = 0;
  for (std::size_t i = 0; i < kRows; ++i) {
    if (h_gpair(i, 0).GetGrad() != 0.0f || h_gpair(i, 0).GetHess() != 0.0f) {
      if (i < kLargeGradRows) {
        large_sampled++;
      } else {
        small_sampled++;
      }
    }
  }

  // Large gradient rows should have higher sampling rate
  double large_ratio = static_cast<double>(large_sampled) / static_cast<double>(kLargeGradRows);
  double small_ratio =
      static_cast<double>(small_sampled) / static_cast<double>(kRows - kLargeGradRows);

  // Large gradient rows should be sampled at a higher rate
  EXPECT_GT(large_ratio, small_ratio * 2);
}

/**
 * @brief Test that adaptive MVS lambda affects sampling behavior.
 *
 * Lambda controls the tradeoff between gradient importance and sample size distribution:
 * - Low lambda (e.g., 0.01): More emphasis on gradient magnitude (importance sampling)
 * - High lambda (e.g., 1.0): More emphasis on hessian, closer to uniform
 */
TEST(CPUGradientBasedSampler, AdaptiveLambda) {
  std::size_t constexpr kRows = 2048;
  double constexpr kSubsample = 0.5;
  Context ctx;

  // Create gradients with large gradient but small hessian in first half,
  // and small gradient but large hessian in second half
  std::size_t shape[2] = {kRows, std::size_t{1}};

  auto run_with_lambda = [&](float lambda) -> std::pair<std::size_t, std::size_t> {
    TrainParam param;
    param.UpdateAllowUnknown(
        Args{{"subsample", std::to_string(kSubsample)}, {"sampling_method", "gradient_based"}});
    param.mvs_adaptive_lambda = lambda;

    linalg::Matrix<GradientPair> gpair(shape, DeviceOrd::CPU());
    auto h_gpair = gpair.HostView();

    for (std::size_t i = 0; i < kRows; ++i) {
      if (i < kRows / 2) {
        h_gpair(i, 0) = GradientPair{1.0f, 0.1f};  // Large g, small h
      } else {
        h_gpair(i, 0) = GradientPair{0.1f, 1.0f};  // Small g, large h
      }
    }

    SampleGradient(&ctx, param, h_gpair);

    // Count sampled rows in each half
    std::size_t first_half_sampled = 0;
    std::size_t second_half_sampled = 0;
    for (std::size_t i = 0; i < kRows; ++i) {
      if (h_gpair(i, 0).GetGrad() != 0.0f || h_gpair(i, 0).GetHess() != 0.0f) {
        if (i < kRows / 2) {
          first_half_sampled++;
        } else {
          second_half_sampled++;
        }
      }
    }
    return {first_half_sampled, second_half_sampled};
  };

  // With low lambda, gradient dominates, first half should be sampled more
  auto [low_lambda_first, low_lambda_second] = run_with_lambda(0.01f);

  // With high lambda, hessian dominates, second half should be sampled more
  auto [high_lambda_first, high_lambda_second] = run_with_lambda(10.0f);

  // Low lambda should favor first half (large gradient)
  EXPECT_GT(low_lambda_first, low_lambda_second);

  // High lambda should favor second half (large hessian)
  EXPECT_GT(high_lambda_second, high_lambda_first);
}
}  // namespace xgboost::tree
