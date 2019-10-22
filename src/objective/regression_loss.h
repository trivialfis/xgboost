/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#ifndef XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_
#define XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_

#include <dmlc/omp.h>
#include <xgboost/logging.h>
#include <xgboost/data.h>

#include <algorithm>
#include "../common/math.h"
#include "../common/quantile.h"
#include "../common/selection.h"
#include "xgboost/span.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace obj {

// common regressions
// linear regression
struct LinearSquareLoss {
  // duplication is necessary, as __device__ specifier
  // cannot be made conditional on template parameter
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return x; }
  XGBOOST_DEVICE static bool CheckLabel(bst_float x) { return true; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    return predt - label;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    return 1.0f;
  }
  template <typename T>
  static T PredTransform(T x) { return x; }
  template <typename T>
  static T FirstOrderGradient(T predt, T label) { return predt - label; }
  template <typename T>
  static T SecondOrderGradient(T predt, T label) { return T(1.0f); }
  static bst_float ProbToMargin(bst_float base_score) { return base_score; }
  static const char* LabelErrorMsg() { return ""; }
  static const char* DefaultEvalMetric() { return "rmse"; }

  static const char* Name() { return "reg:squarederror"; }

  static bool HasHessian() { return true; }
  static float ReestimatePrediction(MetaInfo const& info, HostDeviceVector<float>* predts,
                                    common::Span<size_t const> rids) { return 0; }
};

struct SquaredLogError {
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return x; }
  XGBOOST_DEVICE static bool CheckLabel(bst_float label) {
    return label > -1;
  }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    predt = fmaxf(predt, -1 + 1e-6);  // ensure correct value for log1p
    return (std::log1p(predt) - std::log1p(label)) / (predt + 1);
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    predt = fmaxf(predt, -1 + 1e-6);
    float res = (-std::log1p(predt) + std::log1p(label) + 1) /
                std::pow(predt + 1, 2);
    res = fmaxf(res, 1e-6f);
    return res;
  }
  static bst_float ProbToMargin(bst_float base_score) { return base_score; }
  static const char* LabelErrorMsg() {
    return "label must be greater than -1 for rmsle so that log(label + 1) can be valid.";
  }
  static const char* DefaultEvalMetric() { return "rmsle"; }

  static const char* Name() { return "reg:squaredlogerror"; }

  static bool HasHessian() { return true; }
  static float ReestimatePrediction(MetaInfo const& info, HostDeviceVector<float>* predts,
                                    common::Span<size_t const> rids) { return 0; }
};

// logistic loss for probability regression task
struct LogisticRegression {
  // duplication is necessary, as __device__ specifier
  // cannot be made conditional on template parameter
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return common::Sigmoid(x); }
  XGBOOST_DEVICE static bool CheckLabel(bst_float x) { return x >= 0.0f && x <= 1.0f; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    return predt - label;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const float eps = 1e-16f;
    return fmaxf(predt * (1.0f - predt), eps);
  }
  template <typename T>
  static T PredTransform(T x) { return common::Sigmoid(x); }
  template <typename T>
  static T FirstOrderGradient(T predt, T label) { return predt - label; }
  template <typename T>
  static T SecondOrderGradient(T predt, T label) {
    const T eps = T(1e-16f);
    return std::max(predt * (T(1.0f) - predt), eps);
  }
  static bst_float ProbToMargin(bst_float base_score) {
    CHECK(base_score > 0.0f && base_score < 1.0f)
        << "base_score must be in (0,1) for logistic loss, got: " << base_score;
    return -logf(1.0f / base_score - 1.0f);
  }
  static const char* LabelErrorMsg() {
    return "label must be in [0,1] for logistic regression";
  }
  static const char* DefaultEvalMetric() { return "rmse"; }

  static const char* Name() { return "reg:logistic"; }

  static bool HasHessian() { return true; }
  static float ReestimatePrediction(MetaInfo const& info, HostDeviceVector<float>* predts,
                                    common::Span<size_t const> rids) { return 0; }
};

// logistic loss for binary classification task
struct LogisticClassification : public LogisticRegression {
  static const char* DefaultEvalMetric() { return "error"; }
  static const char* Name() { return "binary:logistic"; }
};

// logistic loss, but predict un-transformed margin
struct LogisticRaw : public LogisticRegression {
  // duplication is necessary, as __device__ specifier
  // cannot be made conditional on template parameter
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return x; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    predt = common::Sigmoid(predt);
    return predt - label;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    const float eps = 1e-16f;
    predt = common::Sigmoid(predt);
    return fmaxf(predt * (1.0f - predt), eps);
  }
  template <typename T>
    static T PredTransform(T x) { return x; }
  template <typename T>
    static T FirstOrderGradient(T predt, T label) {
    predt = common::Sigmoid(predt);
    return predt - label;
  }
  template <typename T>
    static T SecondOrderGradient(T predt, T label) {
    const T eps = T(1e-16f);
    predt = common::Sigmoid(predt);
    return std::max(predt * (T(1.0f) - predt), eps);
  }
  static const char* DefaultEvalMetric() { return "auc"; }

  static const char* Name() { return "binary:logitraw"; }
  static float ReestimatePrediction(MetaInfo const& info, HostDeviceVector<float>* predts,
                                    common::Span<size_t const> rids) { return 0; }
};

struct AbsError {
  XGBOOST_DEVICE static bst_float PredTransform(bst_float x) { return x; }
  XGBOOST_DEVICE static bool CheckLabel(bst_float x) { return true; }
  XGBOOST_DEVICE static bst_float FirstOrderGradient(bst_float predt, bst_float label) {
    bst_float residue = predt - label;
    // The original gradient.
    bst_float gradient = (0 < residue) - (residue < 0);
    return gradient;
  }
  XGBOOST_DEVICE static bst_float SecondOrderGradient(bst_float predt, bst_float label) {
    return 1.0f;
  }
  template <typename T>
  static T PredTransform(T x) { return x; }
  template <typename T>
  static T FirstOrderGradient(T predt, T label) { return predt - label; }
  template <typename T>
  static T SecondOrderGradient(T predt, T label) { return T(1.0f); }
  static bst_float ProbToMargin(bst_float base_score) { return base_score; }
  static const char* LabelErrorMsg() { return ""; }
  static const char* DefaultEvalMetric() { return "mae"; }

  static const char* Name() { return "reg:abs"; }

  static bool HasHessian() { return false; }
  static float ReestimatePrediction(MetaInfo const& info, HostDeviceVector<float>* predts,
                                    common::Span<size_t const> rids) {
    auto const& h_predts = common::Span<float const>{
      predts->ConstHostVector().data(), predts->ConstHostVector().size()};
    auto const& h_labels = common::Span<float const>{
      info.labels_.ConstHostVector().data(), info.labels_.ConstHostVector().size()};
    CHECK_EQ(h_predts.size(), h_labels.size());
    std::vector<float> weighted_residue (rids.size());

#pragma omp parallel for if (rids.size() > 1e4)
    for (size_t i = 0; i < rids.size(); ++i) {
      auto const rid = rids[i];
      weighted_residue[i] = (h_labels[rid] - h_predts[rid]) * info.GetWeight(rid);
    }

    size_t const median_ind = weighted_residue.size() / 2;
    Select(0, weighted_residue.begin(),
           (weighted_residue.begin() + median_ind),
           weighted_residue.end());
    return weighted_residue.at(median_ind);
  }
};

}  // namespace obj
}  // namespace xgboost

#endif  // XGBOOST_OBJECTIVE_REGRESSION_LOSS_H_
