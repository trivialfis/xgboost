#ifndef XGBOOST_TREE_HIST_ROW_PARTITIONER_H_
#define XGBOOST_TREE_HIST_ROW_PARTITIONER_H_

#include "xgboost/base.h"
#include "xgboost/tree_model.h"
#include "../../common/column_matrix.h"
#include "evaluate_splits.h"

namespace xgboost {
namespace tree {

// split row indexes (rid_span) to 2 parts (left_part, right_part) depending
// on comparison of indexes values (idx_span) and split point (split_cond)
// Handle dense columns
// Analog of std::stable_partition, but in no-inplace manner
template <bool default_left, bool any_missing, typename BinIdxType>
inline std::pair<size_t, size_t>
PartitionDenseKernel(const common::DenseColumn<BinIdxType> &column,
                     common::Span<const size_t> rid_span,
                     const int32_t split_cond, common::Span<size_t> left_part,
                     common::Span<size_t> right_part) {
  const int32_t offset = column.GetBaseIdx();
  const BinIdxType* idx = column.GetFeatureBinIdxPtr().data();
  size_t* p_left_part = left_part.data();
  size_t* p_right_part = right_part.data();
  size_t nleft_elems = 0;
  size_t nright_elems = 0;

  if (any_missing) {
    for (auto rid : rid_span) {
      if (column.IsMissing(rid)) {
        if (default_left) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      } else {
        if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
          p_left_part[nleft_elems++] = rid;
        } else {
          p_right_part[nright_elems++] = rid;
        }
      }
    }
  } else {
    for (auto rid : rid_span)  {
      if ((static_cast<int32_t>(idx[rid]) + offset) <= split_cond) {
        p_left_part[nleft_elems++] = rid;
      } else {
        p_right_part[nright_elems++] = rid;
      }
    }
  }
  return {nleft_elems, nright_elems};
}

template <typename Pred>
inline std::pair<size_t, size_t>
PartitionRangeKernel(common::Span<const size_t> ridx, common::Span<size_t> left_part,
                     common::Span<size_t> right_part, Pred pred) {
  size_t* p_left_part = left_part.data();
  size_t* p_right_part = right_part.data();
  size_t nleft_elems = 0;
  size_t nright_elems = 0;
  for (auto row_id : ridx) {
    if (pred(row_id)) {
      p_left_part[nleft_elems++] = row_id;
    } else {
      p_right_part[nright_elems++] = row_id;
    }
  }
  return {nleft_elems, nright_elems};
}

// Split row indexes (rid_span) to 2 parts (left_part, right_part) depending
// on comparison of indexes values (idx_span) and split point (split_cond).
// Handle sparse columns
template<bool default_left, typename BinIdxType>
inline std::pair<size_t, size_t> PartitionSparseKernel(
  common::Span<const size_t> rid_span, const int32_t split_cond,
  const common::SparseColumn<BinIdxType>& column, common::Span<size_t> left_part,
  common::Span<size_t> right_part) {
  size_t* p_left_part  = left_part.data();
  size_t* p_right_part = right_part.data();

  size_t nleft_elems = 0;
  size_t nright_elems = 0;
  const size_t* row_data = column.GetRowData();
  const size_t column_size = column.Size();
  if (rid_span.size()) {  // ensure that rid_span is nonempty range
    // search first nonzero row with index >= rid_span.front()
    const size_t* p = std::lower_bound(row_data, row_data + column_size,
                                       rid_span.front());

    if (p != row_data + column_size && *p <= rid_span.back()) {
      size_t cursor = p - row_data;

      for (auto rid : rid_span) {
        while (cursor < column_size
               && column.GetRowIdx(cursor) < rid
               && column.GetRowIdx(cursor) <= rid_span.back()) {
          ++cursor;
        }
        if (cursor < column_size && column.GetRowIdx(cursor) == rid) {
          if (static_cast<int32_t>(column.GetGlobalBinIdx(cursor)) <= split_cond) {
            p_left_part[nleft_elems++] = rid;
          } else {
            p_right_part[nright_elems++] = rid;
          }
          ++cursor;
        } else {
          // missing value
          if (default_left) {
            p_left_part[nleft_elems++] = rid;
          } else {
            p_right_part[nright_elems++] = rid;
          }
        }
      }
    } else {  // all rows in rid_span have missing values
      if (default_left) {
        std::copy(rid_span.begin(), rid_span.end(), p_left_part);
        nleft_elems = rid_span.size();
      } else {
        std::copy(rid_span.begin(), rid_span.end(), p_right_part);
        nright_elems = rid_span.size();
      }
    }
  }

  return {nleft_elems, nright_elems};
}

// The builder is required for samples partition to left and rights children for set of nodes
// Responsible for:
// 1) Effective memory allocation for intermediate results for multi-thread work
// 2) Merging partial results produced by threads into original row set (row_set_collection_)
// BlockSize is template to enable memory alignment easily with C++11 'alignas()' feature
template<size_t BlockSize>
class PartitionBuilder {
 public:
  template<typename Func>
  void Init(const size_t n_tasks, size_t n_nodes, Func funcNTaks) {
    left_right_nodes_sizes_.resize(n_nodes);
    blocks_offsets_.resize(n_nodes+1);

    blocks_offsets_[0] = 0;
    for (size_t i = 1; i < n_nodes+1; ++i) {
      blocks_offsets_[i] = blocks_offsets_[i-1] + funcNTaks(i-1);
    }

    if (n_tasks > max_n_tasks_) {
      mem_blocks_.resize(n_tasks);
      max_n_tasks_ = n_tasks;
    }
  }

  common::Span<size_t> GetLeftBuffer(int nid, size_t begin, size_t end) {
    const size_t task_idx = GetTaskIdx(nid, begin);
    return { mem_blocks_.at(task_idx).Left(), end - begin };
  }

  common::Span<size_t> GetRightBuffer(int nid, size_t begin, size_t end) {
    const size_t task_idx = GetTaskIdx(nid, begin);
    return { mem_blocks_.at(task_idx).Right(), end - begin };
  }

  void SetNLeftElems(int nid, size_t begin, size_t end, size_t n_left) {
    size_t task_idx = GetTaskIdx(nid, begin);
    mem_blocks_.at(task_idx).n_left = n_left;
  }

  void SetNRightElems(int nid, size_t begin, size_t end, size_t n_right) {
    size_t task_idx = GetTaskIdx(nid, begin);
    mem_blocks_.at(task_idx).n_right = n_right;
  }


  size_t GetNLeftElems(int nid) const {
    return left_right_nodes_sizes_[nid].first;
  }

  size_t GetNRightElems(int nid) const {
    return left_right_nodes_sizes_[nid].second;
  }

  // Each thread has partial results for some set of tree-nodes
  // The function decides order of merging partial results into final row set
  void CalculateRowOffsets() {
    for (size_t i = 0; i < blocks_offsets_.size()-1; ++i) {
      size_t n_left = 0;
      for (size_t j = blocks_offsets_[i]; j < blocks_offsets_[i+1]; ++j) {
        mem_blocks_[j].n_offset_left = n_left;
        n_left += mem_blocks_[j].n_left;
      }
      size_t n_right = 0;
      for (size_t j = blocks_offsets_[i]; j < blocks_offsets_[i+1]; ++j) {
        mem_blocks_[j].n_offset_right = n_left + n_right;
        n_right += mem_blocks_[j].n_right;
      }
      left_right_nodes_sizes_[i] = {n_left, n_right};
    }
  }

  void MergeToArray(int nid, size_t begin, size_t* rows_indexes) const {
    size_t task_idx = GetTaskIdx(nid, begin);

    size_t* left_result  = rows_indexes + mem_blocks_[task_idx].n_offset_left;
    size_t* right_result = rows_indexes + mem_blocks_[task_idx].n_offset_right;

    const size_t* left = mem_blocks_[task_idx].Left();
    const size_t* right = mem_blocks_[task_idx].Right();

    std::copy_n(left, mem_blocks_[task_idx].n_left, left_result);
    std::copy_n(right, mem_blocks_[task_idx].n_right, right_result);
  }

  template <typename Pred>
  void PartitionRange(const size_t node_in_set, const size_t nid,
                      common::Range1d range, bst_feature_t fidx,
                      common::RowSetCollection *p_row_set_collection,
                      Pred pred) {
    auto &row_set_collection = *p_row_set_collection;
    const size_t *p_ridx = row_set_collection[nid].begin;
    common::Span<const size_t> ridx(p_ridx + range.begin(), p_ridx + range.end());
    common::Span<size_t> left =
        this->GetLeftBuffer(node_in_set, range.begin(), range.end());
    common::Span<size_t> right =
        this->GetRightBuffer(node_in_set, range.begin(), range.end());
    std::pair<size_t, size_t> child_nodes_sizes =
        PartitionRangeKernel(ridx, left, right, pred);

    const size_t n_left = child_nodes_sizes.first;
    const size_t n_right = child_nodes_sizes.second;

    this->SetNLeftElems(node_in_set, range.begin(), range.end(), n_left);
    this->SetNRightElems(node_in_set, range.begin(), range.end(), n_right);
  }

  template <typename BinIdxType>
  void PartitionKernel(const size_t node_in_set, const size_t nid,
                       common::Range1d const &range, const int32_t split_cond,
                       const common::ColumnMatrix &column_matrix,
                       const RegTree &tree,
                       common::RowSetCollection *p_row_set_collection) {
    auto &row_set_collection = *p_row_set_collection;
    const size_t *rid = row_set_collection[nid].begin;

    common::Span<const size_t> rid_span(rid + range.begin(), rid + range.end());
    common::Span<size_t> left =
        this->GetLeftBuffer(node_in_set, range.begin(), range.end());
    common::Span<size_t> right =
        this->GetRightBuffer(node_in_set, range.begin(), range.end());
    const bst_uint fid = tree[nid].SplitIndex();
    const bool default_left = tree[nid].DefaultLeft();
    const auto column_ptr = column_matrix.GetColumn<BinIdxType>(fid);

    std::pair<size_t, size_t> child_nodes_sizes;

    if (column_ptr->GetType() == xgboost::common::kDenseColumn) {
      const common::DenseColumn<BinIdxType> &column =
          static_cast<const common::DenseColumn<BinIdxType> &>(
              *(column_ptr.get()));
      if (default_left) {
        if (column_matrix.AnyMissing()) {
          child_nodes_sizes = PartitionDenseKernel<true, true>(
              column, rid_span, split_cond, left, right);
        } else {
          child_nodes_sizes = PartitionDenseKernel<true, false>(
              column, rid_span, split_cond, left, right);
        }
      } else {
        if (column_matrix.AnyMissing()) {
          child_nodes_sizes = PartitionDenseKernel<false, true>(
              column, rid_span, split_cond, left, right);
        } else {
          child_nodes_sizes = PartitionDenseKernel<false, false>(
              column, rid_span, split_cond, left, right);
        }
      }
    } else {
      const common::SparseColumn<BinIdxType> &column =
          static_cast<const common::SparseColumn<BinIdxType> &>(
              *(column_ptr.get()));
      if (default_left) {
        child_nodes_sizes = PartitionSparseKernel<true>(rid_span, split_cond,
                                                        column, left, right);
      } else {
        child_nodes_sizes = PartitionSparseKernel<false>(rid_span, split_cond,
                                                         column, left, right);
      }
    }

    const size_t n_left = child_nodes_sizes.first;
    const size_t n_right = child_nodes_sizes.second;

    this->SetNLeftElems(node_in_set, range.begin(), range.end(), n_left);
    this->SetNRightElems(node_in_set, range.begin(), range.end(), n_right);
  }

 protected:
  size_t GetTaskIdx(int nid, size_t begin) const {
    return blocks_offsets_[nid] + begin / BlockSize;
  }

  struct BlockInfo{
    size_t n_left;
    size_t n_right;

    size_t n_offset_left;
    size_t n_offset_right;

    size_t *Left() { return &left_data_[0]; }
    size_t *Right() { return &right_data_[0]; }

    size_t const* Left() const {
      return &left_data_[0];
    }

    size_t const* Right() const {
      return &right_data_[0];
    }
   private:
    alignas(128) size_t left_data_[BlockSize];
    alignas(128) size_t right_data_[BlockSize];
  };
  std::vector<std::pair<size_t, size_t>> left_right_nodes_sizes_;
  std::vector<size_t> blocks_offsets_;
  std::vector<BlockInfo> mem_blocks_;
  size_t max_n_tasks_ = 0;
};
};  // namespace tree
};  // namespace xgboost

#endif  // XGBOOST_TREE_HIST_ROW_PARTITIONER_H_
