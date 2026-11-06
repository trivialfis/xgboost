/**
 * Copyright 2018-2026, XGBoost Contributors
 * \file split_evaluator.h
 * \brief Used for implementing a loss term specific to decision trees. Useful for custom regularisation.
 * \author Henry Gouk
 */

#ifndef XGBOOST_TREE_SPLIT_EVALUATOR_H_
#define XGBOOST_TREE_SPLIT_EVALUATOR_H_

#include <xgboost/base.h>

#include <cstddef>      // for size_t
#include <limits>       // for numeric_limits
#include <tuple>        // for tuple
#include <type_traits>  // for void_t, false_type, declval

#include "../common/math.h"
#include "../common/transform.h"
#include "param.h"       // for TrainParam
#include "split_gain.h"  // for CalcWeight, CalcGain, CalcGainGivenWeight, IsValidSplit
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost::tree {
namespace split_impl {
template <typename T, typename = void>
struct IsVectorGradientSum : std::false_type {};

template <typename T>
struct IsVectorGradientSum<T,
                           std::void_t<decltype(std::declval<T const&>().Size()),
                                       decltype(std::declval<T const&>()(std::size_t{}).GetGrad()),
                                       decltype(std::declval<T const&>()(std::size_t{}).GetHess())>>
    : std::true_type {};

template <typename... T>
using EnableVecGrad = std::enable_if_t<(split_impl::IsVectorGradientSum<T>::value && ...), int>;

template <typename... T>
using EnableScaGrad = std::enable_if_t<!(split_impl::IsVectorGradientSum<T>::value || ...), int>;
}  // namespace split_impl

struct EvalParam {
  // minimum amount of hessian(weight) allowed in a child
  float min_child_weight;
  // L2 regularization factor
  float reg_lambda;
  // L1 regularization factor
  float reg_alpha;
  // maximum delta update we can add in weight estimation
  // this parameter can be used to stabilize update
  // default=0 means no constraint on weight delta
  float max_delta_step;
  std::uint32_t max_cat_to_onehot;
  bst_bin_t max_cat_threshold;

  EvalParam() = default;

  XGBOOST_DEVICE explicit EvalParam(const TrainParam& param)
      : min_child_weight(param.min_child_weight),
        reg_lambda(param.reg_lambda),
        reg_alpha(param.reg_alpha),
        max_delta_step(param.max_delta_step),
        max_cat_to_onehot{param.max_cat_to_onehot},
        max_cat_threshold{param.max_cat_threshold} {}
};

class TreeEvaluator {
  HostDeviceVector<float> lower_bounds_;
  HostDeviceVector<float> upper_bounds_;
  HostDeviceVector<int32_t> monotone_;
  DeviceOrd device_;
  bst_target_t n_targets_;
  bool has_constraint_;

  void EnsureBounds(bst_node_t max_nidx, bst_target_t n_targets) {
    auto n_nodes = static_cast<std::size_t>(max_nidx) * 2 + 1;
    auto n = n_nodes * n_targets;
    if (lower_bounds_.Size() < n) {
      lower_bounds_.Resize(n, -std::numeric_limits<float>::max());
    }
    if (upper_bounds_.Size() < n) {
      upper_bounds_.Resize(n, std::numeric_limits<float>::max());
    }
  }

 public:
  TreeEvaluator(TrainParam const& p, bst_feature_t n_features, DeviceOrd device,
                bst_target_t n_targets)
      : device_{device}, n_targets_{n_targets}, has_constraint_{p.HasMonotone()} {
    CHECK_GT(n_targets, 0);
    if (device.IsCUDA()) {
      lower_bounds_.SetDevice(device);
      upper_bounds_.SetDevice(device);
      monotone_.SetDevice(device);
    }

    if (!p.monotone_constraints.empty()) {
      CHECK_LE(p.monotone_constraints.size(), n_features)
          << "The size of monotone constraint should be less or equal to the number of features.";
    }
    monotone_.HostVector() = p.monotone_constraints;
    monotone_.HostVector().resize(n_features, 0);

    if (has_constraint_) {
      // Initialised to some small size, can grow if needed
      lower_bounds_.Resize(256 * n_targets, -std::numeric_limits<float>::max());
      upper_bounds_.Resize(256 * n_targets, std::numeric_limits<float>::max());
    }

    if (device_.IsCUDA()) {
      // Pull to device early.
      lower_bounds_.ConstDeviceSpan();
      upper_bounds_.ConstDeviceSpan();
      monotone_.ConstDeviceSpan();
    }
  }

  template <typename ParamT>
  struct SplitEvaluator {
    const int* constraints;
    const float* lower;  // shape: (n_nodes, n_targets)
    const float* upper;
    bool has_constraint;
    bst_target_t n_targets;

    [[nodiscard]] XGBOOST_DEVICE float ApplyBounds(bst_node_t nidx, bst_target_t target,
                                                   float w) const {
      if (!has_constraint) {
        return w;
      }
      std::size_t strides[2]{n_targets, 1};
      auto idx = linalg::detail::Offset<0>(strides, 0, nidx, target);
      if (w < lower[idx]) {
        return lower[idx];
      } else if (w > upper[idx]) {
        return upper[idx];
      } else {
        return w;
      }
    }

    // See the sphinx document about monotone constraint for how this works.
    template <typename LeftGradientSumT, typename RightGradientSumT,
              split_impl::EnableScaGrad<LeftGradientSumT, RightGradientSumT> = 0>
    [[nodiscard]] XGBOOST_DEVICE auto CalcPooledWeight(ParamT const& param, bst_node_t nidx,
                                                       bst_target_t target,
                                                       LeftGradientSumT const& left,
                                                       RightGradientSumT const& right) const {
      auto pooled_weight = ::xgboost::tree::CalcPooledWeight(
          param, left.GetGrad() + right.GetGrad(), left.GetHess() + right.GetHess());
      return this->ApplyBounds(nidx, target, pooled_weight);
    }

    [[nodiscard]] XGBOOST_DEVICE bool CheckConstraint(bst_feature_t fidx, float left,
                                                      float right) const {
      auto constraint = has_constraint ? constraints[fidx] : 0;
      return constraint == 0 || (constraint > 0 && left <= right) ||
             (constraint < 0 && left >= right);
    }

    template <typename LeftGradientSumT, typename RightGradientSumT,
              split_impl::EnableScaGrad<LeftGradientSumT, RightGradientSumT> = 0>
    [[nodiscard]] XGBOOST_DEVICE std::tuple<float, float> CalcSplitWeights(
        ParamT const& param, bst_node_t nidx, bst_feature_t fidx, bst_target_t target,
        LeftGradientSumT const& left, RightGradientSumT const& right) const {
      auto wleft = this->CalcWeight(nidx, target, param, left);
      auto wright = this->CalcWeight(nidx, target, param, right);

      if (this->CheckConstraint(fidx, wleft, wright)) {
        return {wleft, wright};
      }

      auto pooled = this->CalcPooledWeight(param, nidx, target, left, right);
      return {pooled, pooled};
    }

    template <typename LeftGradientSumT, typename RightGradientSumT>
    XGBOOST_DEVICE void AccumulateConstrainedSplitGain(ParamT const& param, bst_node_t nidx,
                                                       bst_feature_t fidx, bst_target_t target,
                                                       LeftGradientSumT const& left,
                                                       RightGradientSumT const& right, bool scalar,
                                                       double* gain) const {
      float l_w, r_w;
      if (scalar) {
        l_w = this->CalcWeight(nidx, target, param, left);
        r_w = this->CalcWeight(nidx, target, param, right);
        // Pooling a violating scalar split would make the weights equal, so reject it
        // before computing the pooled weight. Equal weights cannot improve on the parent.
        if (l_w == r_w || !this->CheckConstraint(fidx, l_w, r_w)) {
          *gain = -std::numeric_limits<double>::infinity();
          return;
        }
      } else {
        auto [left_weight, right_weight] =
            this->CalcSplitWeights(param, nidx, fidx, target, left, right);
        l_w = left_weight;
        r_w = right_weight;
      }
      *gain += tree::CalcGainGivenWeight(param, left.GetGrad(), left.GetHess(), l_w);
      *gain += tree::CalcGainGivenWeight(param, right.GetGrad(), right.GetHess(), r_w);
    }

    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcSplitGain(ParamT const& param, bst_node_t nidx, bst_feature_t fidx,
                                       GradientSumT const& left, GradientSumT const& right) const {
      constexpr float kNegInf = -std::numeric_limits<float>::infinity();
      auto left_hess = left.GetHess();
      auto right_hess = right.GetHess();
      if (!IsValidSplit(param, left_hess, right_hess)) {
        return kNegInf;
      }

      if (has_constraint) {
        double gain{0.0};
        this->AccumulateConstrainedSplitGain(param, nidx, fidx, 0, left, right, true, &gain);
        return gain;
      }
      float wleft = this->CalcWeight(nidx, param, left);
      float wright = this->CalcWeight(nidx, param, right);

      return this->CalcGainGivenWeight(param, left, wleft) +
             this->CalcGainGivenWeight(param, right, wright);
    }

    template <typename LeftGradientSumT, typename RightGradientSumT,
              split_impl::EnableVecGrad<LeftGradientSumT, RightGradientSumT> = 0>
    XGBOOST_DEVICE double CalcSplitGain(ParamT const& param, bst_node_t nidx, bst_feature_t fidx,
                                        LeftGradientSumT const& left,
                                        RightGradientSumT const& right) const {
      auto n_targets = left.Size();
      double left_hess{0.0};
      double right_hess{0.0};
      double gain{0.0};
      for (std::size_t t = 0; t < n_targets; ++t) {
        auto const left_t = left(t);
        auto const right_t = right(t);
        left_hess += left_t.GetHess();
        right_hess += right_t.GetHess();
        if (!has_constraint) {
          gain += tree::CalcGain(param, left_t.GetGrad(), left_t.GetHess());
          gain += tree::CalcGain(param, right_t.GetGrad(), right_t.GetHess());
        }
      }

      auto k = static_cast<double>(n_targets);
      if (!IsValidSplit(param, left_hess / k, right_hess / k)) {
        return -std::numeric_limits<double>::infinity();
      }
      if (!has_constraint) {
        return gain;
      }

      for (std::size_t t = 0; t < n_targets; ++t) {
        this->AccumulateConstrainedSplitGain(param, nidx, fidx, t, left(t), right(t),
                                             n_targets == 1, &gain);
      }
      return gain;
    }

    // Weight
    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcWeight(bst_node_t nidx, bst_target_t target, ParamT const& param,
                                    GradientSumT const& stats) const {
      // boxed by max_delta_step
      float w = ::xgboost::tree::CalcWeight(param, stats);
      return this->ApplyBounds(nidx, target, w);
    }

    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcWeight(bst_node_t nidx, ParamT const& param,
                                    GradientSumT const& stats) const {
      return this->CalcWeight(nidx, 0, param, stats);
    }

    template <typename GradientSumT, split_impl::EnableVecGrad<GradientSumT> = 0>
    XGBOOST_DEVICE void CalcWeight(bst_node_t nidx, ParamT const& param, GradientSumT const& stats,
                                   linalg::VectorView<float> out) const {
      for (std::size_t t = 0; t < stats.Size(); ++t) {
        out(t) = this->CalcWeight(nidx, t, param, stats(t));
      }
    }

    template <typename LeftGradientSumT, typename RightGradientSumT,
              split_impl::EnableVecGrad<LeftGradientSumT, RightGradientSumT> = 0>
    XGBOOST_DEVICE void CalcSplitWeights(ParamT const& param, bst_node_t nidx, bst_feature_t fidx,
                                         LeftGradientSumT const& left,
                                         RightGradientSumT const& right,
                                         linalg::VectorView<float> left_weight,
                                         linalg::VectorView<float> right_weight) const {
      for (std::size_t t = 0; t < left.Size(); ++t) {
        auto [l_w, r_w] = this->CalcSplitWeights(param, nidx, fidx, t, left(t), right(t));
        left_weight(t) = l_w;
        right_weight(t) = r_w;
      }
    }

    /**
     * @brief Unregularized weight -G/H used to sort categories for partition-based splits.
     *
     * Regularization and weight bounds apply to each child as a whole, not to individual
     * categories. With a single target, some optimal partition is a threshold on -G/H.
     */
    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcWeightCat(GradientSumT const& stats) const {
      // Categories absent from the node have empty bins. Avoid NaN keys, which break sorting.
      if (stats.GetHess() <= 0) {
        return 0.0f;
      }
      return static_cast<float>(-stats.GetGrad() / stats.GetHess());
    }

    // Gain given weight
    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcGainGivenWeight(ParamT const& p, GradientSumT const& stats,
                                             float w) const {
      if (stats.GetHess() <= 0) {
        return .0f;
      }
      return tree::CalcGainGivenWeight<ParamT>(p, stats.GetGrad(), stats.GetHess(), w);
    }

    template <typename GradientSumT, typename WeightT, split_impl::EnableVecGrad<GradientSumT> = 0>
    XGBOOST_DEVICE double CalcGainGivenWeight(ParamT const& p, GradientSumT const& stats,
                                              WeightT const& weight) const {
      double gain{0.0};
      for (std::size_t t = 0; t < stats.Size(); ++t) {
        auto const stats_t = stats(t);
        gain += tree::CalcGainGivenWeight(p, stats_t.GetGrad(), stats_t.GetHess(), weight(t));
      }
      return gain;
    }

    // Gain
    template <typename GradientSumT, split_impl::EnableScaGrad<GradientSumT> = 0>
    XGBOOST_DEVICE float CalcGain(bst_node_t nidx, ParamT const& p,
                                  GradientSumT const& stats) const {
      return this->CalcGainGivenWeight(p, stats, this->CalcWeight(nidx, p, stats));
    }

    template <typename GradientSumT, split_impl::EnableVecGrad<GradientSumT> = 0>
    XGBOOST_DEVICE double CalcGain(bst_node_t nidx, ParamT const& p,
                                   GradientSumT const& stats) const noexcept {
      auto for_each_t = [&](auto&& calc_gain) {
        double gain{0.0};
        for (std::size_t t = 0, n_targets = stats.Size(); t < n_targets; ++t) {
          auto const& stats_t = stats(t);
          gain += calc_gain(t, stats_t);
        }
        return gain;
      };

      if (!has_constraint) {
        // Match CalcSplitGain's scoring to avoid spurious positive split gains.
        return for_each_t([&p](std::size_t, auto const& stats_t) {
          return tree::CalcGain(p, stats_t.GetGrad(), stats_t.GetHess());
        });
      }
      return for_each_t([this, nidx, &p](std::size_t t, auto const& stats_t) {
        auto weight = this->CalcWeight(nidx, t, p, stats_t);
        return tree::CalcGainGivenWeight(p, stats_t.GetGrad(), stats_t.GetHess(), weight);
      });
    }
  };

 public:
  /* Get a view to the evaluator that can be passed down to device. */
  template <typename ParamT = TrainParam>
  auto GetEvaluator() const {
    if (device_.IsCUDA()) {
      auto constraints = monotone_.ConstDevicePointer();
      return SplitEvaluator<ParamT>{constraints, lower_bounds_.ConstDevicePointer(),
                                    upper_bounds_.ConstDevicePointer(), has_constraint_,
                                    n_targets_};
    } else {
      auto constraints = monotone_.ConstHostPointer();
      return SplitEvaluator<ParamT>{constraints, lower_bounds_.ConstHostPointer(),
                                    upper_bounds_.ConstHostPointer(), has_constraint_, n_targets_};
    }
  }

  [[nodiscard]] bool HasConstraint() const { return has_constraint_; }

  /** @brief Node IDs and unscaled child weights for one split and target. */
  struct SplitBounds {
    bst_node_t nidx;
    bst_node_t left_nidx;
    bst_node_t right_nidx;
    bst_feature_t fidx;
    float left_weight;
    float right_weight;
  };

  /**
   * @brief Propagate monotonic bounds for a batch of independent splits.
   *
   * The accessor returns SplitBounds for (split index, target index) on the evaluator's
   * device. Parents must already have bounds, children must be distinct, and no child
   * may also be a parent in this batch. max_nidx is the maximum child node ID.
   */
  template <bool CompiledWithCuda = WITH_CUDA(), typename GetSplit>
  void AddSplits(std::size_t n_splits, bst_node_t max_nidx, GetSplit get_split) {
    if (!has_constraint_ || n_splits == 0) {
      return;
    }
    auto n_targets = n_targets_;
    this->EnsureBounds(max_nidx, n_targets);
    common::Transform<CompiledWithCuda>::Init(
        [=] XGBOOST_DEVICE(std::size_t i, common::Span<float> lower, common::Span<float> upper,
                           common::Span<int> monotone) {
          auto t = i % n_targets;
          auto split = get_split(i / n_targets, t);
          auto parent_idx = static_cast<std::size_t>(split.nidx) * n_targets + t;
          auto left_idx = static_cast<std::size_t>(split.left_nidx) * n_targets + t;
          auto right_idx = static_cast<std::size_t>(split.right_nidx) * n_targets + t;

          lower[left_idx] = lower[parent_idx];
          upper[left_idx] = upper[parent_idx];
          lower[right_idx] = lower[parent_idx];
          upper[right_idx] = upper[parent_idx];

          auto mid = split.left_weight + 0.5f * (split.right_weight - split.left_weight);
          SPAN_CHECK(!common::CheckNAN(mid));
          auto constraint = monotone[split.fidx];
          if (constraint < 0) {
            lower[left_idx] = mid;
            upper[right_idx] = mid;
          } else if (constraint > 0) {
            upper[left_idx] = mid;
            lower[right_idx] = mid;
          }
        },
        common::Range(0, n_splits * n_targets), 1, device_)
        .Eval(&lower_bounds_, &upper_bounds_, &monotone_);
  }
};
}  // namespace xgboost::tree

#endif  // XGBOOST_TREE_SPLIT_EVALUATOR_H_
