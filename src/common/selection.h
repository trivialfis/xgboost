/*!
 * Copyright 2019 by Contributors
 * \file selection.h
 * \brief selection algorithm implementation.
 */
#include <rabit/rabit.h>
#include "xgboost/logging.h"
#include <algorithm>
#include <cmath>

namespace xgboost {

template <class Policy, class RandomIt>
RandomIt Partition(Policy policy, RandomIt first, RandomIt last) noexcept(true) {
  // FIXME(trivialfis): Parallelize it.
  RandomIt pivot_iter { first };
  for (auto it = first + 1; it < last + 1; ++it) {
    if (*it < *first) {
      pivot_iter += 1;
      std::iter_swap(pivot_iter, it);
    }
  }
  // FIXME(trivialfis): Advance to match std::partition.
  return pivot_iter;
}

template <class Policy, class RandomIt>
void SelectImpl(Policy&& policy, RandomIt first, RandomIt const& nth, RandomIt last) noexcept(true) {
  while (true) {
    if ( std::distance(first, last) == 0 ) { return; }
    RandomIt pivot_iter {first + std::distance(first, last) / 2};
    std::iter_swap(first, pivot_iter);
    pivot_iter = Partition(policy, first, last);
    std::iter_swap(pivot_iter, first);

    if (nth == pivot_iter) {
      return;
    } else if (nth < pivot_iter) {
      last = pivot_iter - 1;
    } else {
      first = pivot_iter + 1;
    }
  }
}

/* \brief std::nth_element
 *
 * libstdc++ has a bug that somehow `nth_element` is defined without version guard
 * `__cplusplus >= 201402L`.
 *
 * Also libstdc++ uses introselect with heap and insertion sort for better performance
 */
template <class Policy, class RandomIt>
void Select(Policy&& policy, RandomIt first, RandomIt const& nth, RandomIt last) {
  if (first == last) { return; }
  CHECK(first < last);
  // Believe it or not, it's incredibaly difficult to implement this thing with `right ==
  // size()' ...
  SelectImpl(policy, first, nth, std::prev(last));
}
}  // namespace xgboost
