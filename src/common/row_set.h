/*!
 * Copyright 2017 by Contributors
 * \file row_set.h
 * \brief Quick Utility to compute subset of rows
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_COMMON_ROW_SET_H_
#define XGBOOST_COMMON_ROW_SET_H_

#include <xgboost/data.h>
#include <algorithm>
#include <vector>

#include "xgboost/span.h"

namespace xgboost {
namespace common {

/*! \brief collection of rowset */
class RowSetCollection {
 public:
  /*! \brief data structure to store an instance set, a subset of
   *  rows (instances) associated with a particular node in a decision
   *  tree. */
  struct Elem {
    size_t begin {0};
    size_t end {0};
      // id of node associated with this instance set; -1 means uninitialized
    Elem() = default;
    Elem(const size_t begin_, const size_t end_)
        : begin(begin_), end(end_) {}
    Elem(Elem&& other) : begin{other.begin}, end{other.end} {}
    Elem(Elem const& other) : begin{other.begin}, end{other.end} {}

    Elem& operator=(Elem const&) = default;
    Elem& operator=(Elem&&) = default;

    inline size_t Size() const {
      return end - begin;
    }
  };

  size_t Size(unsigned node_id) {
    return elem_of_each_node_[node_id].Size();
  }

  std::vector<Elem>::const_iterator cbegin() const {  // NOLINT
    return elem_of_each_node_.cbegin();
  }

  std::vector<Elem>::const_iterator cend() const {  // NOLINT
    return elem_of_each_node_.cend();
  }

  /*! \brief return corresponding element set given the node_id */
  common::Span<size_t const> operator[](unsigned node_id) const {
    const Elem e { elem_of_each_node_[node_id] };
    return common::Span<size_t const>{row_indices_.data(), row_indices_.size()}.subspan(
        e.begin, e.Size());
  }
  common::Span<size_t> operator[](unsigned node_id) {
    const Elem e { elem_of_each_node_[node_id] };
    return common::Span<size_t>{row_indices_.data(), row_indices_.size()}.subspan(
        e.begin, e.Size());
  }

  // clear up things
  void Clear() {
    elem_of_each_node_.clear();
  }
  // initialize node id 0->everything
  void Init() {
    CHECK_EQ(elem_of_each_node_.size(), 0U);
    // FIXME(trivialfis): https://github.com/dmlc/xgboost/issues/2800
    const size_t begin = 0;
    const size_t end = row_indices_.size();
    elem_of_each_node_.emplace_back(Elem(begin, end));
  }

  // split rowset into two
  void AddSplit(unsigned node_id,
                size_t iLeft,
                unsigned left_node_id,
                unsigned right_node_id) {
    Elem e = elem_of_each_node_[node_id];

    // CHECK(e.begin != 0);

    size_t begin = e.begin;
    size_t split_pt = begin + iLeft;

    if (left_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize((left_node_id + 1)*2, Elem(0, 0));
    }
    if (right_node_id >= elem_of_each_node_.size()) {
      elem_of_each_node_.resize((right_node_id + 1)*2, Elem(0, 0));
    }

    // elem_of_each_node_.
    elem_of_each_node_[left_node_id] = Elem(begin, split_pt);
    elem_of_each_node_[right_node_id] = Elem(split_pt, e.end);
    elem_of_each_node_[node_id] = Elem(begin, e.end);
  }

  // stores the row indices in the set
  std::vector<size_t> row_indices_;

 private:
  // vector: node_id -> elements
  std::vector<Elem> elem_of_each_node_;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_ROW_SET_H_
