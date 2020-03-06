#include "xgboost/span.h"

namespace xgboost {

template <typename E>

class VecExpression {
 public:
  double operator[](size_t i) const {
    // Delegation to the actual expression type. This avoids dynamic
    // polymorphism (a.k.a. virtual functions in C++)
    return static_cast<E const &>(*this)[i];
  }

  size_t Size() const { return static_cast<E const &>(*this).Size(); }
};

template <typename E1, typename E2>
class VecSum : public VecExpression<VecSum<E1, E2>> {
  E1 const &u_;
  E2 const &v_;

 public:
  VecSum(E1 const &u, E2 const &v) : u_(u), v_(v) {
    assert(u.Size() == v.Size());
  }
  double operator[](size_t i) const { return u_[i] + v_[i]; }
  size_t Size() const { return v_.Size(); }
};

class Vec : public VecExpression<Vec> {
  common::Span<double> elems_;

 public:
  double operator[](size_t i) const { return elems_[i]; }
  double &operator[](size_t i) { return elems_[i]; }
  size_t Size() const { return elems_.size(); }

  // A Vec can be constructed from any VecExpression, forcing its evaluation.
  template <typename E>
  explicit Vec(VecExpression<E> const &expr) : elems_(expr.Size()) {
    for (size_t i = 0; i != expr.Size(); ++i) {
      elems_[i] = expr[i];
    }
  }
};
} // namespace xgboost
