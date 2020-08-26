#include <gtest/gtest.h>
#include "../helpers.h"
#include "../../../src/tree/updater_approx.cc"

namespace xgboost {
namespace tree {

std::shared_ptr<DMatrix> GenerateDMatrix(size_t rows, size_t cols) {
  std::vector<float> data;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      data.emplace_back(static_cast<float>(i));
    }
  }
  return GetDMatrixFromData(data, rows, cols);
}

TEST(Approx, InitData) {
  size_t constexpr kRows = 100, kCols = 100;

  auto gpair = GenerateRandomGradients(kRows, .0f, .1f);
  // auto const& h_gpair = gpair.ConstHostVector();

  TrainParam tparam;
  tparam.UpdateAllowUnknown(Args{});
  CPUHistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(Args{});
  common::Monitor monitor;
  GloablApproxBuilder<double> updater(tparam, hist_param, kCols, &monitor);
  common::GHistIndexMatrix gidx;
  common::ColumnMatrix columns;

  auto m = GenerateDMatrix(kRows, kCols);
  // updater.InitData(m.get(), h_gpair, &gidx, &columns);

  // auto const& cuts = gidx.cut;
  // for (size_t i = 1; i < cuts.Ptrs().size(); ++i) {
  //   ASSERT_EQ(cuts.Ptrs()[i-1] + kRows, cuts.Ptrs()[i]);
  // }

  // ASSERT_EQ(cuts.Values().size(), kRows * kCols);
  // for (size_t i = 0; i < cuts.Values().size(); i += kRows) {
  //   // excluding min val and max val.
  //   for (size_t j = 0; j < kRows - 1; ++j) {
  //     ASSERT_EQ(cuts.Values()[i + j], j + 1);
  //   }
  // }
}

class ApproxForTest : public GloablApproxBuilder<double> {
  using SuperT = GloablApproxBuilder<double>;
  size_t rows_;
  size_t cols_;
  common::Monitor monitor_;

 public:
  ApproxForTest(TrainParam param, CPUHistMakerTrainParam hparam,
                size_t n_rows,
                bst_feature_t n_features)
      : SuperT{param, hparam, n_features, &monitor_}, rows_{n_rows}, cols_{n_features} {}

  void TestBuildRootHistogram(const std::vector<GradientPair> &gpair,
                              common::GHistIndexMatrix const &gidx,
                              bool is_dense,
                              std::vector<bst_node_t> const &nodes_to_build) {
    SuperT::BuildNodeHistogram(gpair, gidx, is_dense, nodes_to_build);
    auto root_hist = histogram_builder_.Histograms()[RegTree::kRoot];
    ASSERT_EQ(root_hist.size(), rows_ * cols_);

    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        EXPECT_EQ(root_hist[i * rows_ + j].GetGrad(), 1.0f);
        EXPECT_EQ(root_hist[i * rows_ + j].GetHess(), 2.0f);
      }
    }
  }
};

std::vector<GradientPair> GenerateConstantGradients(bst_row_t rows, float grad,
                                                    float hess) {
  std::vector<GradientPair> gpair(rows, {grad, hess});
  return gpair;
}

class WithNumThreads {
  int32_t ori_threads_;

 public:
  explicit WithNumThreads(int32_t num_threads) {
    ori_threads_ = omp_get_num_threads();
    omp_set_num_threads(num_threads);
  }
  ~WithNumThreads() {
    omp_set_num_threads(ori_threads_);
  }
};

TEST(Approx, BuildHistogram) {
  WithNumThreads ctx{omp_get_num_procs()};
  size_t constexpr kRows = 100, kCols = 100;
  TrainParam tparam;
  tparam.UpdateAllowUnknown(Args{});
  CPUHistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(Args{});
  ApproxForTest updater(tparam, hist_param, kRows, kCols);

  common::GHistIndexMatrix gidx;
  common::ColumnMatrix columns;

  auto h_gpair = GenerateConstantGradients(kRows, 1.0f, 2.0f);

  auto m = GenerateDMatrix(kRows, kCols);
  // updater.InitData(m.get(), h_gpair, &gidx, &columns);

  // updater.TestBuildRootHistogram(h_gpair, gidx, true, {0});
}

TEST(Approx, EvaluateSplit) {

}

TEST(Approx, UpdatePosition) {

}

TEST(Approx, InitRoot) {

}

TEST(Approx, ApplySplit) {
  size_t constexpr kRows = 100, kCols = 100;
  TrainParam tparam;
  tparam.UpdateAllowUnknown(Args{});
  CPUHistMakerTrainParam hist_param;
  hist_param.UpdateAllowUnknown(Args{});
  ApproxForTest updater(tparam, hist_param, kRows, kCols);

  RegTree tree;
  GradStats left_sum {0.1f, 0.3f}, right_sum {0.2f, 0.4f};
  SplitEntry split{0.3, 2, 0.5, left_sum, right_sum};
  LocalExpandEntry entry{0, tree.GetDepth(0), split};
  updater.ApplySplit(entry, &tree);
  ASSERT_EQ(tree.GetNodes().size(), 3ul);
  ASSERT_EQ(tree.Stat(tree[0].LeftChild()).sum_hess, left_sum.GetHess());
  ASSERT_EQ(tree.Stat(tree[0].RightChild()).sum_hess, right_sum.GetHess());
  ASSERT_EQ(tree[0].SplitCond(), 0.5);
}
}  // namespace tree
}  // namespace xgboost
