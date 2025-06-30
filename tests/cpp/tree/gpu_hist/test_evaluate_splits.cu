/**
 * Copyright 2020-2024, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "../../../../src/tree/gpu_hist/evaluate_splits.cuh"
#include "../../collective/test_worker.h"  // for BaseMGPUTest
#include "../../helpers.h"
#include "../test_evaluate_splits.h"  // TestPartitionBasedSplit

namespace xgboost::tree {
namespace {
auto ZeroParam() {
  auto args = Args{{"min_child_weight", "0"}, {"lambda", "0"}};
  TrainParam tparam;
  tparam.UpdateAllowUnknown(args);
  return tparam;
}

GradientQuantiser DummyRoundingFactor(Context const* ctx) {
  thrust::device_vector<GradientPair> gpair(1);
  gpair[0] = {1000.f, 1000.f};  // Tests should not exceed sum of 1000
  return {ctx, dh::ToSpan(gpair), MetaInfo()};
}
}  // anonymous namespace

thrust::device_vector<GradientPairInt64> ConvertToInteger(Context const* ctx,
                                                          std::vector<GradientPairPrecise> x) {
  auto r = DummyRoundingFactor(ctx);
  std::vector<GradientPairInt64> y(x.size());
  for (std::size_t i = 0; i < x.size(); i++) {
    y[i] = r.ToFixedPoint(GradientPair(x[i]));
  }
  return y;
}

TEST_F(TestCategoricalSplitWithMissing, GPUHistEvaluator) {
  auto ctx = MakeCUDACtx(0);
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};
  GPUTrainingParam param{param_};
  cuts_.cut_ptrs_.SetDevice(ctx.Device());
  cuts_.cut_values_.SetDevice(ctx.Device());
  cuts_.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<GradientPairInt64> feature_histogram{
      ConvertToInteger(&ctx, feature_histogram_)};

  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  auto d_feature_types = dh::ToSpan(feature_types);
  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitInputs input{1, 0, quantiser.ToFixedPoint(parent_sum_), dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts_.cut_ptrs_.ConstDeviceSpan(),
                                          cuts_.cut_values_.ConstDeviceSpan(),
                                          cuts_.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{param_, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};

  evaluator.Reset(&ctx, cuts_, dh::ToSpan(feature_types), feature_set.size(), param_, false);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  ASSERT_EQ(result.thresh, 1);
  this->CheckResult(result.loss_chg, result.findex, result.fvalue, result.is_cat,
                    result.dir == kLeftDir, quantiser.ToFloatingPoint(result.left_sum),
                    quantiser.ToFloatingPoint(result.right_sum));
}

TEST(GpuHist, PartitionBasic) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = std::vector<float>{0.0, 1.0, 2.0};
  cuts.cut_ptrs_.HostVector() = std::vector<uint32_t>{0, 3};
  cuts.min_vals_.HostVector() = std::vector<float>{0.0};
  cuts.cut_ptrs_.SetDevice(ctx.Device());
  cuts.cut_values_.SetDevice(ctx.Device());
  cuts.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  auto max_cat =
      *std::max_element(cuts.cut_values_.HostVector().begin(), cuts.cut_values_.HostVector().end());
  cuts.SetCategorical(true, max_cat);
  d_feature_types = dh::ToSpan(feature_types);
  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitSharedInputs shared_inputs{
      param,
      quantiser,
      d_feature_types,
      cuts.cut_ptrs_.ConstDeviceSpan(),
      cuts.cut_values_.ConstDeviceSpan(),
      cuts.min_vals_.ConstDeviceSpan(),
      false,
  };

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);

  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-5.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-3.0, 1.0}});
    EvaluateSplitInputs input{0, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }

  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-7.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-3.0, 1.0}, {-3.0, 1.0}});
    EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(cats, std::bitset<32>("10000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  {
    // All -1.0, gain from splitting should be 0.0
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-3.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});
    EvaluateSplitInputs input{2, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_FLOAT_EQ(result.loss_chg, 0.0f);
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  // With 3.0/3.0 missing values
  // Forward, first 2 categories are selected, while the last one go to left along with missing
  // value
  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 6.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}});
    EvaluateSplitInputs input{3, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-5.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-3.0, 1.0}, {-1.0, 1.0}});
    EvaluateSplitInputs input{4, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.dir, kLeftDir);
    EXPECT_EQ(cats, std::bitset<32>("10100000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
  {
    // -1.0s go right
    // -3.0s go left
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-5.0, 3.0});
    auto feature_histogram = ConvertToInteger(&ctx, {{-3.0, 1.0}, {-1.0, 1.0}, {-3.0, 1.0}});
    EvaluateSplitInputs input{5, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(cats, std::bitset<32>("01000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
}

TEST(GpuHist, PartitionTwoFeatures) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = std::vector<float>{0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
  cuts.cut_ptrs_.HostVector() = std::vector<uint32_t>{0, 3, 6};
  cuts.min_vals_.HostVector() = std::vector<float>{0.0, 0.0};
  cuts.cut_ptrs_.SetDevice(ctx.Device());
  cuts.cut_values_.SetDevice(ctx.Device());
  cuts.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types(dh::ToSpan(feature_types));
  auto max_cat =
      *std::max_element(cuts.cut_values_.HostVector().begin(), cuts.cut_values_.HostVector().end());
  cuts.SetCategorical(true, max_cat);

  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);

  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-6.0, 3.0});
    auto feature_histogram = ConvertToInteger(
        &ctx, {{-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}});
    EvaluateSplitInputs input{0, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.findex, 1);
    EXPECT_EQ(cats, std::bitset<32>("11000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }

  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-6.0, 3.0});
    auto feature_histogram = ConvertToInteger(
        &ctx, {{-2.0, 1.0}, {-2.0, 1.0}, {-2.0, 1.0}, {-1.0, 1.0}, {-2.5, 1.0}, {-2.5, 1.0}});
    EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                              dh::ToSpan(feature_histogram)};
    DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
    auto cats = std::bitset<32>(evaluator.GetHostNodeCats(input.nidx)[0]);
    EXPECT_EQ(result.findex, 1);
    EXPECT_EQ(cats, std::bitset<32>("10000000000000000000000000000000"));
    EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
  }
}

TEST(GpuHist, PartitionTwoNodes) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  tparam.max_cat_to_onehot = 0;
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts;
  cuts.cut_values_.HostVector() = std::vector<float>{0.0, 1.0, 2.0};
  cuts.cut_ptrs_.HostVector() = std::vector<uint32_t>{0, 3};
  cuts.min_vals_.HostVector() = std::vector<float>{0.0};
  cuts.cut_ptrs_.SetDevice(ctx.Device());
  cuts.cut_values_.SetDevice(ctx.Device());
  cuts.min_vals_.SetDevice(ctx.Device());
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};

  thrust::device_vector<int> monotonic_constraints(feature_set.size(), 0);
  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types(dh::ToSpan(feature_types));
  auto max_cat =
      *std::max_element(cuts.cut_values_.HostVector().begin(), cuts.cut_values_.HostVector().end());
  cuts.SetCategorical(true, max_cat);

  auto quantiser = DummyRoundingFactor(&ctx);
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);

  {
    auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{-6.0, 3.0});
    auto feature_histogram_a = ConvertToInteger(
        &ctx, {{-1.0, 1.0}, {-2.5, 1.0}, {-2.5, 1.0}, {-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}});
    thrust::device_vector<EvaluateSplitInputs> inputs(2);
    inputs[0] = EvaluateSplitInputs{0, 0, parent_sum, dh::ToSpan(feature_set),
                                    dh::ToSpan(feature_histogram_a)};
    auto feature_histogram_b = ConvertToInteger(&ctx, {{-1.0, 1.0}, {-1.0, 1.0}, {-4.0, 1.0}});
    inputs[1] = EvaluateSplitInputs{1, 0, parent_sum, dh::ToSpan(feature_set),
                                    dh::ToSpan(feature_histogram_b)};
    thrust::device_vector<GPUExpandEntry> results(2);
    evaluator.EvaluateSplits(&ctx, {0, 1}, 1, dh::ToSpan(inputs), shared_inputs,
                             dh::ToSpan(results));
    EXPECT_EQ(std::bitset<32>(evaluator.GetHostNodeCats(0)[0]),
              std::bitset<32>("10000000000000000000000000000000"));
    EXPECT_EQ(std::bitset<32>(evaluator.GetHostNodeCats(1)[0]),
              std::bitset<32>("11000000000000000000000000000000"));
  }
}

void TestEvaluateSingleSplit(bool is_categorical) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts{
      MakeCutsForTest({1.0, 2.0, 11.0, 12.0}, {0, 2, 4}, {0.0, 0.0}, ctx.Device())};
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};

  // Setup gradients so that second feature gets higher gain
  auto feature_histogram =
      ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}});

  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  if (is_categorical) {
    auto max_cat = *std::max_element(cuts.cut_values_.HostVector().begin(),
                                     cuts.cut_values_.HostVector().end());
    cuts.SetCategorical(true, max_cat);
    d_feature_types = dh::ToSpan(feature_types);
  }

  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, false);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  if (is_categorical) {
    ASSERT_TRUE(std::isnan(result.fvalue));
  } else {
    EXPECT_EQ(result.fvalue, 11.0);
  }
  EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
}

TEST(GpuHist, EvaluateSingleSplit) { TestEvaluateSingleSplit(false); }

TEST(GpuHist, EvaluateSingleCategoricalSplit) { TestEvaluateSingleSplit(true); }

TEST(GpuHist, EvaluateSingleSplitMissing) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{1.0, 1.5});
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0};
  auto feature_histogram = ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator(tparam, feature_set.size(), FstCU());
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
  EXPECT_EQ(result.dir, kRightDir);
  EXPECT_EQ(result.left_sum, quantiser.ToFixedPoint(GradientPairPrecise(-0.5, 0.5)));
  EXPECT_EQ(result.right_sum, quantiser.ToFixedPoint(GradientPairPrecise(1.5, 1.0)));
}

TEST(GpuHist, EvaluateSingleSplitEmpty) {
  auto ctx = MakeCUDACtx(0);
  TrainParam tparam = ZeroParam();
  GPUHistEvaluator evaluator(tparam, 1, FstCU());
  DeviceSplitCandidate result =
      evaluator
          .EvaluateSingleSplit(
              &ctx, EvaluateSplitInputs{},
              EvaluateSplitSharedInputs{
                  GPUTrainingParam(tparam), DummyRoundingFactor(&ctx), {}, {}, {}, {}, false})
          .split;
  EXPECT_EQ(result.findex, -1);
  EXPECT_LT(result.loss_chg, 0.0f);
}

// Feature 0 has a better split, but the algorithm must select feature 1
TEST(GpuHist, EvaluateSingleSplitFeatureSampling) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{1};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2, 4};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0, 10.0};
  auto feature_histogram =
      ConvertToInteger(&ctx, {{-10.0, 0.5}, {10.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator(tparam, feature_min_values.size(), FstCU());
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  EXPECT_EQ(result.fvalue, 11.0);
  EXPECT_EQ(result.left_sum, quantiser.ToFixedPoint(GradientPairPrecise(-0.5, 0.5)));
  EXPECT_EQ(result.right_sum, quantiser.ToFixedPoint(GradientPairPrecise(0.5, 0.5)));
}

// Features 0 and 1 have identical gain, the algorithm must select 0
TEST(GpuHist, EvaluateSingleSplitBreakTies) {
  auto ctx = MakeCUDACtx(0);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2, 4};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0, 10.0};
  auto feature_histogram =
      ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator(tparam, feature_min_values.size(), FstCU());
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 0);
  EXPECT_EQ(result.fvalue, 1.0);
}

TEST(GpuHist, EvaluateSplits) {
  auto ctx = MakeCUDACtx(0);
  thrust::device_vector<DeviceSplitCandidate> out_splits(2);
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  tparam.UpdateAllowUnknown(Args{});
  GPUTrainingParam param{tparam};

  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};
  thrust::device_vector<uint32_t> feature_segments = std::vector<bst_idx_t>{0, 2, 4};
  thrust::device_vector<float> feature_values = std::vector<float>{1.0, 2.0, 11.0, 12.0};
  thrust::device_vector<float> feature_min_values = std::vector<float>{0.0, 0.0};
  auto feature_histogram_left =
      ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}, {-1.0, 0.5}, {1.0, 0.5}});
  auto feature_histogram_right =
      ConvertToInteger(&ctx, {{-1.0, 0.5}, {1.0, 0.5}, {-0.5, 0.5}, {0.5, 0.5}});
  EvaluateSplitInputs input_left{1, 0, parent_sum, dh::ToSpan(feature_set),
                                 dh::ToSpan(feature_histogram_left)};
  EvaluateSplitInputs input_right{2, 0, parent_sum, dh::ToSpan(feature_set),
                                  dh::ToSpan(feature_histogram_right)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          {},
                                          dh::ToSpan(feature_segments),
                                          dh::ToSpan(feature_values),
                                          dh::ToSpan(feature_min_values),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_min_values.size()),
                             FstCU()};
  dh::device_vector<EvaluateSplitInputs> inputs =
      std::vector<EvaluateSplitInputs>{input_left, input_right};
  evaluator.LaunchEvaluateSplits(input_left.feature_set.size(), dh::ToSpan(inputs), shared_inputs,
                                 evaluator.GetEvaluator(), dh::ToSpan(out_splits));

  DeviceSplitCandidate result_left = out_splits[0];
  EXPECT_EQ(result_left.findex, 1);
  EXPECT_EQ(result_left.fvalue, 11.0);

  DeviceSplitCandidate result_right = out_splits[1];
  EXPECT_EQ(result_right.findex, 0);
  EXPECT_EQ(result_right.fvalue, 1.0);
}

TEST_F(TestPartitionBasedSplit, GpuHist) {
  auto ctx = MakeCUDACtx(0);
  dh::device_vector<FeatureType> ft{std::vector<FeatureType>{FeatureType::kCategorical}};
  GPUHistEvaluator evaluator{param_, static_cast<bst_feature_t>(info_.num_col_), ctx.Device()};

  cuts_.cut_ptrs_.SetDevice(ctx.Device());
  cuts_.cut_values_.SetDevice(ctx.Device());
  cuts_.min_vals_.SetDevice(ctx.Device());

  evaluator.Reset(&ctx, cuts_, dh::ToSpan(ft), info_.num_col_, param_, false);

  // Convert the sample histogram to fixed point
  auto quantiser = DummyRoundingFactor(&ctx);
  thrust::host_vector<GradientPairInt64> h_hist;
  for (auto e : hist_[0]) {
    h_hist.push_back(quantiser.ToFixedPoint(e));
  }
  dh::device_vector<GradientPairInt64> d_hist = h_hist;
  dh::device_vector<bst_feature_t> feature_set{std::vector<bst_feature_t>{0}};

  EvaluateSplitInputs input{0, 0, quantiser.ToFixedPoint(total_gpair_), dh::ToSpan(feature_set),
                            dh::ToSpan(d_hist)};
  EvaluateSplitSharedInputs shared_inputs{GPUTrainingParam{param_},
                                          quantiser,
                                          dh::ToSpan(ft),
                                          cuts_.cut_ptrs_.ConstDeviceSpan(),
                                          cuts_.cut_values_.ConstDeviceSpan(),
                                          cuts_.min_vals_.ConstDeviceSpan(),
                                          false};
  auto split = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
  ASSERT_NEAR(split.loss_chg, best_score_, 1e-2);
}

class MGPUHistTest : public collective::BaseMGPUTest {};

namespace {
void VerifyColumnSplitEvaluateSingleSplit(bool is_categorical) {
  auto ctx = MakeCUDACtx(GPUIDX);
  auto rank = collective::GetRank();
  auto quantiser = DummyRoundingFactor(&ctx);
  auto parent_sum = quantiser.ToFixedPoint(GradientPairPrecise{0.0, 1.0});
  TrainParam tparam = ZeroParam();
  GPUTrainingParam param{tparam};

  common::HistogramCuts cuts{
      rank == 0 ? MakeCutsForTest({1.0, 2.0}, {0, 2, 2}, {0.0, 0.0}, ctx.Device())
                : MakeCutsForTest({11.0, 12.0}, {0, 0, 2}, {0.0, 0.0}, ctx.Device())};
  thrust::device_vector<bst_feature_t> feature_set = std::vector<bst_feature_t>{0, 1};

  // Setup gradients so that second feature gets higher gain
  auto feature_histogram = rank == 0 ? ConvertToInteger(&ctx, {{-0.5, 0.5}, {0.5, 0.5}})
                                     : ConvertToInteger(&ctx, {{-1.0, 0.5}, {1.0, 0.5}});

  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kCategorical);
  common::Span<FeatureType> d_feature_types;
  if (is_categorical) {
    auto max_cat = *std::max_element(cuts.cut_values_.HostVector().begin(),
                                     cuts.cut_values_.HostVector().end());
    cuts.SetCategorical(true, max_cat);
    d_feature_types = dh::ToSpan(feature_types);
  }

  EvaluateSplitInputs input{1, 0, parent_sum, dh::ToSpan(feature_set),
                            dh::ToSpan(feature_histogram)};
  EvaluateSplitSharedInputs shared_inputs{param,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          false};

  GPUHistEvaluator evaluator{tparam, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};
  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), tparam, true);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;

  EXPECT_EQ(result.findex, 1);
  if (is_categorical) {
    ASSERT_TRUE(std::isnan(result.fvalue));
  } else {
    EXPECT_EQ(result.fvalue, 11.0);
  }
  EXPECT_EQ(result.left_sum + result.right_sum, parent_sum);
}
}  // anonymous namespace

TEST_F(MGPUHistTest, ColumnSplitEvaluateSingleSplit) {
  if (curt::AllVisibleGPUs() > 1) {
    // We can't emulate multiple GPUs with NCCL.
    this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(false); }, false, true);
  }
  this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(false); }, true, true);
}

TEST_F(MGPUHistTest, ColumnSplitEvaluateSingleCategoricalSplit) {
  if (curt::AllVisibleGPUs() > 1) {
    // We can't emulate multiple GPUs with NCCL.
    this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(true); }, false, true);
  }
  this->DoTest([] { VerifyColumnSplitEvaluateSingleSplit(true); }, true, true);
}

common::HistogramCuts ReprCuts(Context const* ctx) {
  common::HistogramCuts cuts;
  cuts.cut_ptrs_ = {
      0,     50,    100,   150,   200,   250,   300,   350,   400,   450,   500,   550,   600,
      650,   700,   750,   800,   850,   900,   950,   1000,  1050,  1100,  1150,  1200,  1250,
      1300,  1350,  1400,  1450,  1500,  1550,  1600,  1650,  1700,  1750,  1800,  1850,  1900,
      1950,  2000,  2050,  2100,  2150,  2200,  2250,  2300,  2350,  2400,  2450,  2500,  2550,
      2600,  2650,  2700,  2750,  2800,  2850,  2900,  2950,  3000,  3050,  3100,  3150,  3200,
      3250,  3300,  3350,  3400,  3450,  3500,  3550,  3600,  3650,  3700,  3750,  3800,  3850,
      3900,  3950,  4000,  4050,  4100,  4150,  4200,  4250,  4300,  4350,  4400,  4450,  4500,
      4550,  4600,  4650,  4700,  4750,  4800,  4850,  4900,  4950,  5000,  5050,  5100,  5150,
      5200,  5250,  5300,  5350,  5400,  5450,  5500,  5550,  5600,  5650,  5700,  5750,  5800,
      5850,  5900,  5950,  6000,  6050,  6100,  6150,  6200,  6250,  6300,  6350,  6400,  6450,
      6500,  6550,  6600,  6650,  6700,  6750,  6800,  6850,  6900,  6950,  7000,  7050,  7100,
      7150,  7200,  7250,  7300,  7350,  7400,  7450,  7500,  7550,  7600,  7650,  7700,  7750,
      7800,  7850,  7900,  7950,  8000,  8050,  8100,  8150,  8200,  8250,  8300,  8350,  8400,
      8450,  8500,  8550,  8600,  8650,  8700,  8750,  8800,  8850,  8900,  8950,  9000,  9050,
      9100,  9150,  9200,  9250,  9300,  9350,  9400,  9450,  9500,  9550,  9600,  9650,  9700,
      9750,  9800,  9850,  9900,  9950,  10000, 10050, 10100, 10150, 10200, 10250, 10300, 10350,
      10400, 10450, 10500, 10550, 10600, 10650, 10700, 10750, 10800, 10850, 10900, 10950, 11000,
      11050, 11100, 11150, 11200, 11250, 11300, 11350, 11400, 11450, 11500, 11550, 11600, 11650,
      11700, 11750, 11800, 11850, 11900, 11950, 12000, 12050, 12100, 12150, 12200, 12250, 12300,
      12350, 12400, 12450, 12500, 12550, 12600, 12650, 12700, 12750, 12800, 12850, 12900, 12950,
      13000, 13050, 13100, 13150, 13200, 13250, 13300, 13350, 13400, 13450, 13500, 13550, 13600,
      13650, 13700, 13750, 13800, 13850, 13900, 13950, 14000, 14050, 14100, 14150, 14200, 14250,
      14300, 14350, 14400, 14450, 14500, 14550, 14600, 14650, 14700, 14750, 14800, 14850, 14900,
      14950, 15000, 15050, 15100, 15150, 15200, 15250, 15300, 15350, 15400, 15450, 15500, 15550,
      15600, 15650, 15700, 15750, 15800, 15850, 15900, 15950, 16000, 16050, 16100, 16150, 16200,
      16250, 16300, 16350, 16400, 16450, 16500, 16550, 16600, 16650, 16700, 16750, 16800, 16850,
      16900, 16950, 17000, 17050, 17100, 17150, 17200, 17250, 17300, 17350, 17400, 17450, 17500,
      17550, 17600, 17650, 17700, 17750, 17800, 17850, 17900, 17950, 18000, 18050, 18100, 18150,
      18200, 18250, 18300, 18350, 18400, 18450, 18500, 18550, 18600, 18650, 18700, 18750, 18800,
      18850, 18900, 18950, 19000, 19050, 19100, 19150, 19200, 19250, 19300, 19350, 19400, 19450,
      19500, 19550, 19600, 19650, 19700, 19750, 19800, 19850, 19900, 19950, 20000, 20050, 20100,
      20150, 20200, 20250, 20300, 20350, 20400, 20450, 20500, 20550, 20600, 20650, 20700, 20750,
      20800, 20850, 20900, 20950, 21000, 21050, 21100, 21150, 21200, 21250, 21300, 21350, 21400,
      21450, 21500, 21550, 21600, 21650, 21700, 21750, 21800, 21850, 21900, 21950, 22000, 22050,
      22100, 22150, 22200, 22250, 22300, 22350, 22400, 22450, 22500, 22550, 22600, 22650, 22700,
      22750, 22800, 22850, 22900, 22950, 23000, 23050, 23100, 23150, 23200, 23250, 23300, 23350,
      23400, 23450, 23500, 23550, 23600, 23650, 23700, 23750, 23800, 23850, 23900, 23950, 24000,
      24050, 24100, 24150, 24200, 24250, 24300, 24350, 24400, 24450, 24500, 24550, 24600, 24650,
      24700, 24750, 24800, 24850, 24900, 24950, 25000, 25050, 25100, 25150, 25200, 25250, 25300,
      25350, 25400, 25450, 25500, 25550, 25600
  };
  cuts.min_vals_ = {
      -16.4658, -16.8601, -16.4658, -16.8601, -16.2558, -16.8601, -16.2558, -16.9132, -16.2235,
      -17.3739, -16.2235, -17.3739, -16.2235, -17.3739, -16.2235, -17.3739, -16.2235, -17.3739,
      -17.5295, -17.3739, -17.5295, -17.3739, -17.6294, -17.3739, -17.6294, -17.3739, -17.6294,
      -17.3739, -17.6294, -17.3739, -17.6294, -17.3739, -17.6294, -15.8369, -17.6294, -15.8369,
      -17.6294, -15.8369, -17.6294, -15.8369, -17.6294, -16.2642, -17.6294, -16.2642, -17.6294,
      -16.2642, -17.6294, -16.5586, -15.995,  -16.5586, -15.995,  -16.5586, -16.4191, -16.5586,
      -16.4191, -16.5586, -16.5169, -16.5586, -16.5169, -16.5586, -16.5169, -16.5732, -16.5169,
      -16.5732, -16.5169, -16.6705, -16.5169, -16.6705, -16.5169, -16.7675, -16.5169, -16.7675,
      -16.7265, -16.7675, -16.7265, -16.7675, -16.7265, -16.7675, -16.7265, -16.7675, -16.7265,
      -16.7675, -16.7265, -16.7675, -16.7265, -16.7675, -16.7265, -16.7675, -16.7265, -16.7675,
      -16.7265, -16.7675, -16.7265, -16.7675, -16.7265, -15.6475, -16.3866, -15.4814, -16.3866,
      -15.5371, -16.3866, -15.5371, -16.3866, -15.639,  -16.3866, -15.639,  -16.3866, -15.639,
      -16.3866, -15.639,  -15.6977, -15.639,  -15.7988, -15.639,  -15.7988, -15.639,  -15.8993,
      -15.639,  -15.8993, -16.7466, -17.4458, -16.7466, -17.4458, -16.7466, -17.4458, -16.7466,
      -17.4458, -16.7466, -17.4458, -16.7466, -17.4458, -16.7466, -17.4458, -16.7466, -17.4458,
      -16.7466, -17.4458, -16.7466, -17.4458, -16.7466, -17.4458, -16.7466, -17.4458, -16.1501,
      -17.4458, -16.1501, -17.2036, -16.1501, -17.2036, -16.1501, -17.2036, -16.1501, -17.2036,
      -15.6335, -16.653,  -15.6172, -16.653,  -18.1675, -16.653,  -18.1675, -16.653,  -18.1675,
      -16.653,  -18.1675, -16.653,  -18.1675, -16.653,  -18.1675, -16.3362, -18.1675, -16.3362,
      -18.1675, -16.3362, -18.1675, -16.3362, -18.1675, -16.3362, -18.1675, -16.0914, -18.1675,
      -16.0914, -16.4907, -16.0914, -16.4907, -16.0914, -16.4907, -15.9451, -16.4907, -15.9451,
      -17.0044, -15.9451, -17.0044, -15.9451, -17.0044, -15.9451, -17.0044, -15.9745, -17.0044,
      -15.9745, -17.0044, -15.9745, -17.0044, -15.9745, -17.0044, -15.9745, -17.0044, -15.9745,
      -17.0044, -15.9745, -17.0044, -15.9745, -17.3107, -15.9745, -17.3107, -15.9745, -17.3107,
      -15.9745, -17.3107, -15.9745, -17.3107, -15.7134, -17.3107, -15.9898, -17.3107, -15.9898,
      -17.3107, -15.9898, -17.3107, -15.9898, -17.3107, -15.9898, -17.3107, -15.9898, -17.3107,
      -15.9898, -16.9723, -15.9898, -16.9723, -15.9898, -16.9723, -15.9898, -16.9723, -15.9898,
      -16.9723, -15.9898, -16.9723, -15.9898, -16.299,  -15.8175, -16.299,  -15.8175, -16.3972,
      -15.8175, -16.3972, -16.3558, -16.3972, -16.3558, -16.3972, -16.4538, -16.3972, -16.4538,
      -16.3972, -16.4538, -16.3972, -16.4538, -16.3972, -16.4538, -16.4081, -16.4538, -16.4081,
      -16.4538, -16.4081, -16.4538, -16.4081, -16.4538, -16.4081, -16.4538, -16.4081, -16.4538,
      -16.4081, -16.4538, -16.4081, -16.1103, -16.4081, -16.1103, -16.4081, -16.1103, -16.4081,
      -16.1103, -16.4081, -16.1103, -16.164,  -16.1103, -16.164,  -15.7596, -16.164,  -15.7596,
      -16.164,  -15.7596, -15.9172, -15.7596, -15.9172, -17.0753, -16.0383, -17.0753, -16.0383,
      -17.0753, -16.0383, -17.0753, -16.0383, -17.0753, -16.0383, -17.0753, -16.442,  -17.0753,
      -16.442,  -17.0753, -16.442,  -17.0753, -16.442,  -17.0753, -16.442,  -17.0753, -16.442,
      -17.0753, -16.442,  -17.0753, -16.442,  -16.8354, -16.442,  -16.8354, -16.442,  -16.8354,
      -16.442,  -16.8354, -16.442,  -16.5037, -16.442,  -16.5037, -17.7539, -16.5037, -17.7539,
      -16.5037, -17.7539, -16.2813, -17.7539, -16.2813, -17.7539, -17.9192, -17.7539, -17.9192,
      -17.7539, -17.9192, -17.7539, -17.9192, -17.7539, -17.9192, -17.7539, -17.9192, -17.7539,
      -17.9192, -17.7539, -17.9192, -16.2157, -17.9192, -16.2157, -17.9192, -16.2157, -17.9192,
      -16.2157, -17.9192, -16.6361, -17.9192, -16.6361, -16.0271, -16.6361, -16.0271, -16.6361,
      -16.0271, -16.6361, -16.3267, -16.6361, -16.7892, -16.6361, -16.7892, -16.6361, -16.886,
      -16.6361, -16.886,  -16.6361, -16.886,  -16.6361, -16.886,  -16.6361, -16.886,  -17.0387,
      -16.886,  -17.0387, -16.886,  -17.0387, -16.886,  -17.0387, -16.886,  -17.0387, -16.886,
      -17.0387, -16.886,  -17.0387, -16.886,  -17.0387, -16.0852, -17.0387, -16.0852, -17.0387,
      -15.7951, -17.0387, -15.7951, -17.0387, -15.7951, -17.0387, -15.9645, -16.6037, -15.9645,
      -16.6037, -15.9645, -16.6037, -15.9645, -16.6037, -15.9645, -15.9218, -15.9645, -15.9218,
      -15.9645, -16.0217, -15.9645, -16.0217, -15.9795, -16.0217, -15.9795, -16.0217, -15.9795,
      -16.0217, -15.9795, -16.0217, -15.9795, -16.1365, -15.9795, -16.1365, -15.9795, -17.1147,
      -15.9795, -17.1147, -15.9795, -17.1147, -15.9795, -17.1147, -15.9795, -17.1147, -15.9795,
      -17.1147, -15.7295, -17.1147, -15.7295, -17.1147, -15.7295, -17.1147, -15.7295, -17.1147,
      -15.7295, -17.1147, -15.7295, -17.1147, -15.7295, -16.0162, -15.7295, -16.0162, -15.3714,
      -16.0162, -15.3714, -16.0162, -15.3714, -16.0162, -15.3714, -16.0162, -16.7072, -16.0002,
      -16.7072, -16.0002, -16.7072, -16.0002, -16.7072, -16.0002, -16.7072, -16.0002, -16.7072,
      -16.8601, -16.7072, -16.8601, -16.7072, -16.8601, -16.7072, -16.8601, -16.7072, -16.8601,
      -16.7072, -16.8601, -16.7072, -16.8601, -16.7072, -16.8601, -16.4658, -16.8601};

  std::ifstream fin{"./cut_values"};
  while (!fin.eof()) {
    float v = std::numeric_limits<float>::quiet_NaN();
    fin >> v;
    CHECK(!isnan(v));
    cuts.cut_values_.HostVector().push_back(v);

    auto c = fin.get();
    EXPECT_EQ(c, ',');
    if (fin.eof()) {
      break;
    }
    c = fin.peek();
    if (c == '\n') {
      fin.get();
    } else if(c == -1) {
      EXPECT_EQ(cuts.cut_values_.Size(), 25600);
      break;
    } else {
      EXPECT_EQ(c, ' ') << v << " size:" << cuts.cut_values_.Size();
      fin.get();
    }

  }
  cuts.cut_values_.SetDevice(ctx->Device());
  cuts.cut_values_.ConstDevicePointer();
  cuts.cut_ptrs_.SetDevice(ctx->Device());
  cuts.cut_ptrs_.ConstDevicePointer();
  cuts.min_vals_.SetDevice(ctx->Device());
  cuts.min_vals_.ConstDevicePointer();
  return cuts;
}

TEST(GpuSplit, Input) {
  auto ctx = MakeCUDACtx(0);

  std::vector<GradientPairInt64> h_node_hist;
  std::ifstream fin{"./d_node_hist"};
  while (!fin.eof()) {
    std::int64_t grad = 0, hess = 0;
    fin >> grad;
    auto c = fin.get();
    ASSERT_EQ(c, '/');
    fin >> hess;
    c = fin.get();
    ASSERT_EQ(c, ',');
    c = fin.peek();
    if (c == '\n') {
      fin.get();
    }
    h_node_hist.emplace_back(grad, hess);
  }
  std::cout << h_node_hist.size() << std::endl;
  thrust::device_vector<GradientPairInt64> d_node_hist{h_node_hist};

  thrust::device_vector<bst_feature_t> feature_set(512);
  thrust::sequence(feature_set.begin(), feature_set.end(), 0);

  dh::device_vector<FeatureType> feature_types(feature_set.size(), FeatureType::kNumerical);
  auto d_feature_types = dh::ToSpan(feature_types);

  GradientPairPrecise to_floating_point_{2.32831e-10, 1.45519e-11};
  GradientPairPrecise to_fixed_point{4.29497e+09, 6.87195e+10};
  auto quantiser = GradientQuantiser{to_fixed_point, to_floating_point_};
  TrainParam param;
  param.UpdateAllowUnknown(Args{{"max_depth", "8"}, {"max_bin", "50"}});
  auto gparam = GPUTrainingParam{param};

  auto cuts = ReprCuts(&ctx);

  GradientPairInt64 root_sum{-26782335912, 2882304448711884800};
  EvaluateSplitInputs input{0, 0, root_sum, dh::ToSpan(feature_set), dh::ToSpan(d_node_hist)};
  EvaluateSplitSharedInputs shared_inputs{gparam,
                                          quantiser,
                                          d_feature_types,
                                          cuts.cut_ptrs_.ConstDeviceSpan(),
                                          cuts.cut_values_.ConstDeviceSpan(),
                                          cuts.min_vals_.ConstDeviceSpan(),
                                          true};

  GPUHistEvaluator evaluator{param, static_cast<bst_feature_t>(feature_set.size()), ctx.Device()};

  evaluator.Reset(&ctx, cuts, dh::ToSpan(feature_types), feature_set.size(), param, false);
  DeviceSplitCandidate result = evaluator.EvaluateSingleSplit(&ctx, input, shared_inputs).split;
}
}  // namespace xgboost::tree
