#include "hist/row_partitioner.h"
namespace xgboost {
namespace tree {
class HistRowPartitioner {
  static constexpr size_t kPartitionBlockSize = 2048;
  PartitionBuilder<kPartitionBlockSize> partition_builder_;
  common::RowSetCollection row_set_collection_;

 public:
  void UpdatePosition(common::GHistIndexMatrix const &index,
                      common::ColumnMatrix const& column_matrix,
                      std::vector<LocalExpandEntry> const &candidates,
                      RegTree const *p_tree) {
    size_t n_nodes = candidates.size();
    common::BlockedSpace2d space{n_nodes,
                                 [&](size_t node_in_set) {
                                   auto candidate = candidates[node_in_set];
                                   int32_t nid = candidate.nid;
                                   return row_set_collection_[nid].Size();
                                 },
                                 kPartitionBlockSize};
    partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
      const int32_t nid = candidates[node_in_set].nid;
      const size_t size = row_set_collection_[nid].Size();
      const size_t n_tasks =
          size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
      return n_tasks;
    });
    std::vector<int32_t> split_conditions;
    FindSplitConditions(candidates, *p_tree, index, &split_conditions);
    auto threads = omp_get_max_threads();
    common::ParallelFor2d(
        space, threads, [&](size_t node_in_set, common::Range1d r) {
          const bst_node_t nid = candidates[node_in_set].nid;
          switch (column_matrix.GetTypeSize()) {
          case common::kUint8BinsTypeSize:
            partition_builder_.template PartitionKernel<uint8_t>(
                node_in_set, nid, r, split_conditions[node_in_set],
                column_matrix, *p_tree, &row_set_collection_);
            break;
          case common::kUint16BinsTypeSize:
            partition_builder_.template PartitionKernel<uint16_t>(
                node_in_set, nid, r, split_conditions[node_in_set],
                column_matrix, *p_tree, &row_set_collection_);
            break;
          case common::kUint32BinsTypeSize:
            partition_builder_.template PartitionKernel<uint32_t>(
                node_in_set, nid, r, split_conditions[node_in_set],
                column_matrix, *p_tree, &row_set_collection_);
            break;
          default:
            CHECK(false); // no default behavior
          }
        });

    partition_builder_.CalculateRowOffsets();
    // 4. Copy elements from partition_builder_ to row_set_collection_ back
    // with updated row-indexes for each tree-node
    common::ParallelFor2d(
        space, threads, [&](size_t node_in_set, common::Range1d r) {
          auto candidate = candidates[node_in_set];
          const int32_t nid = candidate.nid;
          partition_builder_.MergeToArray(
              node_in_set, r.begin(),
              const_cast<size_t *>(row_set_collection_[nid].begin));
        });

    for (size_t i = 0; i < n_nodes; ++i) {
      auto nidx = candidates[i].nid;
      auto n_left = partition_builder_.GetNLeftElems(i);
      auto n_right = partition_builder_.GetNRightElems(i);
      CHECK_EQ(n_left + n_right, row_set_collection_[nidx].Size());
      bst_node_t left_nidx = (*p_tree)[nidx].LeftChild();
      bst_node_t right_nidx = (*p_tree)[nidx].RightChild();
      row_set_collection_.AddSplit(nidx, left_nidx, right_nidx, n_left,
                                   n_right);
    }
  }

  auto const& Partitions() const { return row_set_collection_; }
  auto operator[](bst_node_t nidx) { return row_set_collection_[nidx]; }
  size_t Size() const {
    return std::distance(row_set_collection_.begin(),
                         row_set_collection_.end());
  }

  HistRowPartitioner() = default;
  explicit HistRowPartitioner(bst_row_t num_row) {
    row_set_collection_.Clear();
    auto p_positions = row_set_collection_.Data();
    p_positions->resize(num_row);
    std::iota(p_positions->begin(), p_positions->end(), 0);
    row_set_collection_.Init();
  }
};
}  // namespace tree
}  // namespace xgboost
