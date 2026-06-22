/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <numeric>  // for accumulate
#include <vector>   // for vector

#include "../../../src/tree/cv_fold_info.h"

namespace xgboost::tree {
namespace {
// Materialize the global row indices covered by a list of ranges.
std::vector<bst_idx_t> Expand(std::vector<RowRange> const& ranges) {
  std::vector<bst_idx_t> out;
  for (auto const& r : ranges) {
    for (bst_idx_t i = r.first; i < r.second; ++i) {
      out.push_back(i);
    }
  }
  return out;
}
}  // anonymous namespace

TEST(CVFoldInfo, ContiguousLayout) {
  // N divisible by K.
  auto info = CVFoldInfo::MakeContiguous(12, 3);
  ASSERT_EQ(info.n_rows, 12);
  ASSERT_EQ(info.n_folds, 3);
  ASSERT_EQ(info.valid_ptr, (std::vector<bst_idx_t>{0, 4, 8, 12}));

  for (std::int32_t f = 0; f < info.n_folds; ++f) {
    EXPECT_EQ(info.ValidRows(f), 4);
    EXPECT_EQ(info.TrainRows(f), 8);
    // Train and valid partition the whole row set without overlap.
    auto train = Expand(info.TrainRanges(f));
    auto valid = Expand(info.ValidRanges(f));
    EXPECT_EQ(train.size() + valid.size(), info.n_rows);
    std::vector<bst_idx_t> all = train;
    all.insert(all.end(), valid.begin(), valid.end());
    std::sort(all.begin(), all.end());
    std::vector<bst_idx_t> expected(info.n_rows);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(all, expected);
  }
}

TEST(CVFoldInfo, NotDivisible) {
  // N not divisible by K: remainder spread across leading blocks.
  auto info = CVFoldInfo::MakeContiguous(10, 3);
  ASSERT_EQ(info.valid_ptr, (std::vector<bst_idx_t>{0, 3, 6, 10}));
  EXPECT_EQ(info.ValidRows(0), 3);
  EXPECT_EQ(info.ValidRows(1), 3);
  EXPECT_EQ(info.ValidRows(2), 4);
  EXPECT_EQ(info.TrainRows(2), 6);
}

TEST(CVFoldInfo, EdgeRanges) {
  auto info = CVFoldInfo::MakeContiguous(9, 3);  // blocks [0,3) [3,6) [6,9)
  // First fold: validation at the front, so a single training range.
  EXPECT_EQ(info.TrainRanges(0), (std::vector<RowRange>{{3, 9}}));
  // Middle fold: two training ranges.
  EXPECT_EQ(info.TrainRanges(1), (std::vector<RowRange>{{0, 3}, {6, 9}}));
  // Last fold: validation at the back, single training range.
  EXPECT_EQ(info.TrainRanges(2), (std::vector<RowRange>{{0, 6}}));
}

TEST(CVFoldInfo, SingleFold) {
  auto info = CVFoldInfo::MakeContiguous(5, 1);
  EXPECT_EQ(info.ValidRows(0), 5);
  EXPECT_EQ(info.TrainRows(0), 0);
  // The whole dataset is the validation block; no training rows.
  EXPECT_TRUE(info.TrainRanges(0).empty());
  EXPECT_EQ(info.ValidRanges(0), (std::vector<RowRange>{{0, 5}}));
}

TEST(RangesInWindow, Basic) {
  std::vector<RowRange> ranges{{0, 3}, {6, 9}};
  // Window fully inside the gap -> empty.
  EXPECT_TRUE(RangesInWindow(ranges, 3, 6).empty());
  // Window clipping both ranges.
  EXPECT_EQ(RangesInWindow(ranges, 2, 7), (std::vector<RowRange>{{2, 3}, {6, 7}}));
  // Window covering everything.
  EXPECT_EQ(RangesInWindow(ranges, 0, 9), ranges);
}

TEST(MakeFoldBatchView, MultiBatch) {
  // 12 rows, 3 folds (blocks of 4), split into 3 equal pages of 4 rows each.
  auto info = CVFoldInfo::MakeContiguous(12, 3);
  std::vector<bst_idx_t> batch_ptr{0, 4, 8, 12};

  // Fold 0 validates on [0, 4); page 0 is fully validation -> inactive.
  auto v0 = MakeFoldBatchView(info, 0, batch_ptr);
  EXPECT_EQ(v0.per_batch_rows, (std::vector<bst_idx_t>{0, 4, 4}));
  EXPECT_EQ(v0.active_batches, (std::vector<std::int32_t>{1, 2}));
  EXPECT_EQ(v0.TotalRows(), info.TrainRows(0));
  EXPECT_TRUE(v0.per_batch_runs[0].empty());
  EXPECT_EQ(v0.per_batch_runs[1], (std::vector<RowRange>{{4, 8}}));

  // Fold 1 validates on [4, 8); the middle page is fully validation.
  auto v1 = MakeFoldBatchView(info, 1, batch_ptr);
  EXPECT_EQ(v1.per_batch_rows, (std::vector<bst_idx_t>{4, 0, 4}));
  EXPECT_EQ(v1.active_batches, (std::vector<std::int32_t>{0, 2}));
  EXPECT_EQ(v1.TotalRows(), info.TrainRows(1));
}

TEST(MakeFoldBatchView, ValidationStraddlesPages) {
  // 10 rows, 2 folds (blocks [0,5) [5,10)); pages of size 4,3,3.
  auto info = CVFoldInfo::MakeContiguous(10, 2);
  std::vector<bst_idx_t> batch_ptr{0, 4, 7, 10};

  // Fold 0 validates on [0, 5): page 0 [0,4) fully validation, page 1 [4,7) partially.
  auto v0 = MakeFoldBatchView(info, 0, batch_ptr);
  EXPECT_EQ(v0.per_batch_rows, (std::vector<bst_idx_t>{0, 2, 3}));
  EXPECT_EQ(v0.active_batches, (std::vector<std::int32_t>{1, 2}));
  EXPECT_EQ(v0.per_batch_runs[1], (std::vector<RowRange>{{5, 7}}));
  EXPECT_EQ(v0.TotalRows(), info.TrainRows(0));

  // Fold 1 validates on [5, 10): training is [0,5).
  auto v1 = MakeFoldBatchView(info, 1, batch_ptr);
  EXPECT_EQ(v1.per_batch_rows, (std::vector<bst_idx_t>{4, 1, 0}));
  EXPECT_EQ(v1.active_batches, (std::vector<std::int32_t>{0, 1}));
  EXPECT_EQ(v1.per_batch_runs[1], (std::vector<RowRange>{{4, 5}}));
  EXPECT_EQ(v1.TotalRows(), info.TrainRows(1));
}
}  // namespace xgboost::tree
