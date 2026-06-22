/**
 * Copyright 2026, XGBoost Contributors
 *
 * @brief Lightweight description of a K-fold cross-validation layout.
 *
 * Used by the fused CV training path (GPU `hist` + external memory) to describe, for `K`
 * folds over `N` rows, which global rows belong to each fold's validation block and which
 * belong to its training set. The POC supports the "basic" contiguous-block layout: the
 * dataset is partitioned into K contiguous row blocks and fold `f` validates on block `f`
 * and trains on the remaining `K - 1` blocks. Any shuffling is the caller's responsibility
 * and must be reflected in the row ordering of the DMatrix.
 */
#ifndef XGBOOST_TREE_CV_FOLD_INFO_H_
#define XGBOOST_TREE_CV_FOLD_INFO_H_

#include <algorithm>  // for max, min
#include <cstdint>    // for int32_t
#include <utility>    // for pair
#include <vector>     // for vector

#include "xgboost/base.h"     // for bst_idx_t
#include "xgboost/logging.h"  // for CHECK

namespace xgboost::tree {
/**
 * @brief A contiguous half-open global row interval `[begin, end)`.
 */
using RowRange = std::pair<bst_idx_t, bst_idx_t>;

/**
 * @brief Cross-validation fold layout (contiguous-block split).
 */
struct CVFoldInfo {
  /** @brief Total number of rows in the shared DMatrix. */
  bst_idx_t n_rows{0};
  /** @brief Number of folds. */
  std::int32_t n_folds{0};
  /**
   * @brief Global validation block boundaries, size `n_folds + 1`.
   *
   * Validation block of fold `f` is `[valid_ptr[f], valid_ptr[f + 1])`.
   */
  std::vector<bst_idx_t> valid_ptr;

  CVFoldInfo() = default;

  /**
   * @brief Build a contiguous-block layout for `n_folds` folds over `n_rows` rows.
   *
   * Block `f` is `[floor(f * N / K), floor((f + 1) * N / K))`, which spreads any remainder
   * across the leading blocks so block sizes differ by at most one. `K` need not divide
   * `N`.
   */
  static CVFoldInfo MakeContiguous(bst_idx_t n_rows, std::int32_t n_folds) {
    CHECK_GE(n_folds, 1) << "Number of folds must be positive.";
    CHECK_GE(n_rows, static_cast<bst_idx_t>(n_folds))
        << "Number of rows must be at least the number of folds.";
    CVFoldInfo info;
    info.n_rows = n_rows;
    info.n_folds = n_folds;
    info.valid_ptr.resize(static_cast<std::size_t>(n_folds) + 1);
    for (std::int32_t f = 0; f <= n_folds; ++f) {
      info.valid_ptr[f] = (n_rows * static_cast<bst_idx_t>(f)) / static_cast<bst_idx_t>(n_folds);
    }
    CHECK_EQ(info.valid_ptr.front(), 0);
    CHECK_EQ(info.valid_ptr.back(), n_rows);
    return info;
  }

  /** @brief Number of validation rows of fold `f`. */
  [[nodiscard]] bst_idx_t ValidRows(std::int32_t f) const {
    this->CheckFold(f);
    return valid_ptr[f + 1] - valid_ptr[f];
  }
  /** @brief Number of training rows of fold `f`. */
  [[nodiscard]] bst_idx_t TrainRows(std::int32_t f) const { return n_rows - this->ValidRows(f); }

  /** @brief Validation interval of fold `f` (always a single contiguous range). */
  [[nodiscard]] std::vector<RowRange> ValidRanges(std::int32_t f) const {
    this->CheckFold(f);
    if (valid_ptr[f] == valid_ptr[f + 1]) {
      return {};
    }
    return {{valid_ptr[f], valid_ptr[f + 1]}};
  }

  /**
   * @brief Training intervals of fold `f` (everything except the validation block).
   *
   * For the contiguous layout this is the block before the validation block and the block
   * after it, so at most two ranges (fewer when the validation block is at an edge or
   * empty).
   */
  [[nodiscard]] std::vector<RowRange> TrainRanges(std::int32_t f) const {
    this->CheckFold(f);
    std::vector<RowRange> ranges;
    if (valid_ptr[f] > 0) {
      ranges.emplace_back(bst_idx_t{0}, valid_ptr[f]);
    }
    if (valid_ptr[f + 1] < n_rows) {
      ranges.emplace_back(valid_ptr[f + 1], n_rows);
    }
    return ranges;
  }

 private:
  void CheckFold(std::int32_t f) const {
    CHECK_GE(f, 0);
    CHECK_LT(f, n_folds);
    CHECK_EQ(valid_ptr.size(), static_cast<std::size_t>(n_folds) + 1);
  }
};

/**
 * @brief Intersect a set of disjoint, sorted global ranges with a window `[win_begin,
 *        win_end)`.
 *
 * Returns the clipped sub-ranges (in global indices), preserving order and dropping empty
 * intersections. Used to compute the per-batch training-row runs of a fold given a page's
 * global row window.
 */
[[nodiscard]] inline std::vector<RowRange> RangesInWindow(std::vector<RowRange> const& ranges,
                                                          bst_idx_t win_begin, bst_idx_t win_end) {
  std::vector<RowRange> out;
  for (auto const& r : ranges) {
    auto begin = std::max(r.first, win_begin);
    auto end = std::min(r.second, win_end);
    if (begin < end) {
      out.emplace_back(begin, end);
    }
  }
  return out;
}

/**
 * @brief Per-batch training-row view of a single fold over a batched (paged) DMatrix.
 *
 * @param batch_ptr Prefix-sum of page sizes, size `n_batches + 1`; page `k` covers global
 *                  rows `[batch_ptr[k], batch_ptr[k + 1])`.
 */
struct FoldBatchView {
  /** @brief For each source batch `k`, the fold's training-row runs (global indices). */
  std::vector<std::vector<RowRange>> per_batch_runs;
  /** @brief Source batch indices that contain at least one training row of the fold. */
  std::vector<std::int32_t> active_batches;
  /** @brief Per-batch training-row count (size `n_batches`). */
  std::vector<bst_idx_t> per_batch_rows;

  [[nodiscard]] bst_idx_t TotalRows() const {
    bst_idx_t n = 0;
    for (auto v : per_batch_rows) {
      n += v;
    }
    return n;
  }
};

/**
 * @brief Compute the per-batch training-row view of fold `f`.
 */
[[nodiscard]] inline FoldBatchView MakeFoldBatchView(CVFoldInfo const& folds, std::int32_t f,
                                                     std::vector<bst_idx_t> const& batch_ptr) {
  CHECK_GE(batch_ptr.size(), 2);
  CHECK_EQ(batch_ptr.front(), 0);
  CHECK_EQ(batch_ptr.back(), folds.n_rows);
  auto train = folds.TrainRanges(f);
  std::size_t n_batches = batch_ptr.size() - 1;
  FoldBatchView view;
  view.per_batch_runs.resize(n_batches);
  view.per_batch_rows.resize(n_batches, 0);
  for (std::size_t k = 0; k < n_batches; ++k) {
    auto runs = RangesInWindow(train, batch_ptr[k], batch_ptr[k + 1]);
    bst_idx_t n = 0;
    for (auto const& r : runs) {
      n += r.second - r.first;
    }
    view.per_batch_rows[k] = n;
    view.per_batch_runs[k] = std::move(runs);
    if (n > 0) {
      view.active_batches.push_back(static_cast<std::int32_t>(k));
    }
  }
  return view;
}
}  // namespace xgboost::tree

#endif  // XGBOOST_TREE_CV_FOLD_INFO_H_
