/**
 * Copyright 2026, XGBoost Contributors
 */
#include "gradient_based_sampler.h"  // for kDefaultMvsLambda

#include <algorithm>  // for copy, min
#include <cmath>      // for sqrt
#include <cstddef>    // for size_t
#include <numeric>    // for partial_sum
#include <random>     // for default_random_engine, uniform_real_distribution
#include <vector>     // for vector

#include "../../common/algorithm.h"  // for Sort
#include "../../common/math.h"       // for Sqr
#include "../../common/random.h"     // for GlobalRandom
#include "xgboost/base.h"            // for GradientPair, GradientPairPrecise
#include "xgboost/linalg.h"          // for MatrixView
#include "xgboost/span.h"            // for Span

namespace xgboost::tree::cpu_impl {

/**
 * @brief Calculate the threshold μ and find the appropriate threshold index.
 *
 * The threshold μ is found such that the expected sample rate equals the desired rate:
 * E[sample_rate] = (1/μ) * sum(ĝ_i for ĝ_i < μ) + count(ĝ_i >= μ) = sample_rows
 *
 * For threshold μ in (sorted_rag[i], sorted_rag[i+1]]:
 * - Elements 0..i have p = ĝ/μ < 1
 * - Elements i+1..n-1 have p = 1
 * - Expected samples = grad_csum[i]/μ + (n_rows - i - 1) = sample_rows
 * - Therefore: μ = grad_csum[i] / (sample_rows - n_rows + i + 1)
 *
 * @param sorted_rag Sorted regularized absolute gradients (ascending order)
 * @param grad_csum Cumulative sum of sorted gradients
 * @param n_rows Total number of rows
 * @param sample_rows Target number of samples
 * @return The computed threshold μ
 */
float CalculateThreshold(common::Span<float const> sorted_rag, common::Span<float const> grad_csum,
                         std::size_t n_rows, std::size_t sample_rows) {
  // Search for the correct interval (sorted_rag[i], sorted_rag[i+1]] that contains μ
  for (std::size_t i = 0; i < n_rows; ++i) {
    float lower = sorted_rag[i];
    // Upper bound is next element or infinity for last element
    float upper = (i + 1 < n_rows) ? sorted_rag[i + 1] : std::numeric_limits<float>::max();

    // Number of elements above threshold (elements i+1..n-1 have p=1)
    std::size_t n_above = n_rows - i - 1;
    float denom = static_cast<float>(sample_rows) - static_cast<float>(n_above);
    if (denom <= 0) {
      continue;  // Would need to sample more than remaining rows above
    }

    float u = grad_csum[i] / denom;
    if (u > lower && u <= upper) {
      return u;
    }
  }
  // Fallback: use the cumulative sum divided by sample_rows
  return grad_csum[n_rows - 1] / static_cast<float>(sample_rows);
}

void GradientBasedSample(Context const* ctx, linalg::MatrixView<GradientPair> gpairs,
                         float subsample, float mvs_lambda) {
  std::size_t n_rows = gpairs.Shape(0);
  std::size_t n_targets = gpairs.Shape(1);
  std::size_t sample_rows = static_cast<std::size_t>(n_rows * subsample);

  if (sample_rows >= n_rows) {
    return;  // No sampling needed
  }

  // Use default lambda if adaptive lambda is not set or invalid
  float lambda = (mvs_lambda > 0.0f) ? mvs_lambda : kDefaultMvsLambda;

  // Step 1: Calculate regularized absolute gradient for each row
  // For multi-target, sum the squared gradients across targets before taking sqrt
  std::vector<float> reg_abs_grad(n_rows);
  for (std::size_t i = 0; i < n_rows; ++i) {
    float sum_sq = 0.0f;
    for (std::size_t t = 0; t < n_targets; ++t) {
      auto [g, h] = std::make_pair(gpairs(i, t).GetGrad(), gpairs(i, t).GetHess());
      sum_sq += common::Sqr(g) + lambda * common::Sqr(h);
    }
    reg_abs_grad[i] = std::sqrt(sum_sq);
  }

  // Step 2: Sort gradients and compute cumulative sum
  std::vector<float> sorted_rag = reg_abs_grad;  // Copy for sorting
  common::Sort(ctx, sorted_rag.begin(), sorted_rag.end(), std::less{});

  std::vector<float> grad_csum(n_rows);
  std::partial_sum(sorted_rag.begin(), sorted_rag.end(), grad_csum.begin());

  // Step 3: Find threshold using linear search (could be optimized to binary search)
  float threshold = CalculateThreshold(
      common::Span<float const>{sorted_rag.data(), sorted_rag.size()},
      common::Span<float const>{grad_csum.data(), grad_csum.size()}, n_rows, sample_rows);

  // Step 4: Sample rows using Poisson sampling
  auto& rnd = common::GlobalRandom();
  std::size_t seed = rnd();
  std::default_random_engine eng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (std::size_t i = 0; i < n_rows; ++i) {
    float combined_gradient = reg_abs_grad[i];
    float p = std::min(combined_gradient / threshold, 1.0f);

    // Skip rows with zero gradient (already zero)
    if (combined_gradient == 0.0f) {
      continue;
    }

    if (p >= 1.0f) {
      // Always select this row, no scaling needed
      continue;
    }

    float rand_val = dist(eng);
    if (rand_val <= p) {
      // Selected: scale gradient by 1/p
      float scale = 1.0f / p;
      for (std::size_t t = 0; t < n_targets; ++t) {
        auto old = gpairs(i, t);
        gpairs(i, t) = GradientPair{old.GetGrad() * scale, old.GetHess() * scale};
      }
    } else {
      // Not selected: zero out
      for (std::size_t t = 0; t < n_targets; ++t) {
        gpairs(i, t) = GradientPair{};
      }
    }
  }
}
}  // namespace xgboost::tree::cpu_impl
