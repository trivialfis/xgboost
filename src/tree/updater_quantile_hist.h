/*!
 * Copyright 2017-2019 by Contributors
 * \file updater_quantile_hist.h
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Chen, Egor Smirnov
 */
#ifndef XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
#define XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_

#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <iomanip>
#include <unordered_map>
#include <utility>
#include <tuple>

#include "./param.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"

namespace xgboost {
namespace common {
  struct GradStatHist;
}
namespace tree {

using xgboost::common::GHistIndexMatrix;
using xgboost::common::GHistIndexBlockMatrix;
using xgboost::common::GHistIndexRow;
using xgboost::common::HistCollection;
using xgboost::common::RowSetCollection;
using xgboost::common::GHistRow;
using xgboost::common::GHistBuilder;
using xgboost::common::ColumnMatrix;
using xgboost::common::Column;

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker: public TreeUpdater {
 public:
  void Configure(const Args& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix* data,
                             HostDeviceVector<bst_float>* out_preds) override;

  char const* Name() const override {
    return "grow_quantile_histmaker";
  }

 protected:
  // training parameter
  TrainParam param_;
  // quantized data matrix
  GHistIndexMatrix gmat_;
  // (optional) data matrix with feature grouping
  GHistIndexBlockMatrix gmatb_;
  // column accessor
  ColumnMatrix column_matrix_;
  bool is_gmat_initialized_;

  // data structure
 public:
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
    /*! \brief weight calculated related to current data */
    float weight;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit NodeEntry(const TrainParam& param)
        : root_gain(0.0f), weight(0.0f) {
    }
  };
  // actual builder that runs the algorithm

  struct Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner,
                     std::unique_ptr<SplitEvaluator> spliteval,
                     LeaveIndexCache* index_cache)
      : param_(param), pruner_(std::move(pruner)), spliteval_(std::move(spliteval)),
        p_last_tree_(nullptr), p_last_fmat_(nullptr), index_cache_{index_cache} {
      perf_monitor.Init("QuantileHistMaker::Builder");
    }
    // update one tree, growing
    virtual void Update(const GHistIndexMatrix& gmat,
                        const GHistIndexBlockMatrix& gmatb,
                        const ColumnMatrix& column_matrix,
                        HostDeviceVector<GradientPair>* gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree);

    bool UpdatePredictionCache(const DMatrix* data,
                               HostDeviceVector<bst_float>* p_out_preds);

    std::tuple<common::GradStatHist::GradType*, common::GradStatHist*>
    GetHistBuffer(std::vector<uint8_t>* hist_is_init,
      std::vector<common::GradStatHist>* grad_stats, size_t block_id, size_t nthread,
      size_t tid, std::vector<common::GradStatHist::GradType*>* data_hist, size_t hist_size);

   protected:
    /* tree growing policies */
    struct ExpandEntry {
      int nid;
      int sibling_nid;
      int parent_nid;
      int depth;
      bst_float loss_chg;
      unsigned timestamp;
      ExpandEntry(int nid, int sibling_nid, int parent_nid, int depth, bst_float loss_chg,
        unsigned tstmp) : nid(nid), sibling_nid(sibling_nid), parent_nid(parent_nid),
        depth(depth), loss_chg(loss_chg), timestamp(tstmp) {}
    };

    // initialize temp data structure
    void InitData(const GHistIndexMatrix& gmat,
                  const std::vector<GradientPair>& gpair,
                  const DMatrix& fmat,
                  const RegTree& tree);

    void InitNewNode(int nid,
                     const GHistIndexMatrix& gmat,
                     const std::vector<GradientPair>& gpair,
                     const DMatrix& fmat,
                     RegTree* tree,
                     QuantileHistMaker::NodeEntry* snode,
                     int32_t parentid);

    // enumerate the split values of specific feature
    bool EnumerateSplit(int d_step,
                        const GHistIndexMatrix& gmat,
                        const GHistRow& hist,
                        const NodeEntry& snode,
                        const MetaInfo& info,
                        SplitEntry* p_best,
                        bst_uint fid,
                        bst_uint nodeID);

    void EvaluateSplitsBatch(const std::vector<ExpandEntry>& nodes,
          const GHistIndexMatrix& gmat,
          const DMatrix& fmat,
          const std::vector<std::vector<uint8_t>>& hist_is_init,
          const std::vector<std::vector<common::GradStatHist::GradType*>>& hist_buffers);

    void ReduceHistograms(
        common::GradStatHist::GradType* hist_data,
        common::GradStatHist::GradType* sibling_hist_data,
        common::GradStatHist::GradType* parent_hist_data,
        const size_t ibegin,
        const size_t iend,
        const size_t inode,
        const std::vector<std::vector<uint8_t>>& hist_is_init,
        const std::vector<std::vector<common::GradStatHist::GradType*>>& hist_buffers);

    void SyncHistograms(
        RegTree* p_tree,
        const std::vector<ExpandEntry>& nodes,
        std::vector<std::vector<common::GradStatHist::GradType*>>* hist_buffers,
        std::vector<std::vector<uint8_t>>* hist_is_init,
        const std::vector<std::vector<common::GradStatHist>>& grad_stats);

     void ExpandWithDepthWise(const GHistIndexMatrix &gmat,
                              const GHistIndexBlockMatrix &gmatb,
                              const ColumnMatrix &column_matrix,
                              DMatrix *p_fmat,
                              RegTree *p_tree,
                              const std::vector<GradientPair> &gpair_h);


    void ExpandWithLossGuide(const GHistIndexMatrix& gmat,
                             const GHistIndexBlockMatrix& gmatb,
                             const ColumnMatrix& column_matrix,
                             DMatrix* p_fmat,
                             RegTree* p_tree,
                             const std::vector<GradientPair>& gpair_h);


    void BuildHistsBatch(const std::vector<ExpandEntry>& nodes, RegTree* tree,
      const GHistIndexMatrix &gmat, const std::vector<GradientPair>& gpair,
      std::vector<std::vector<common::GradStatHist::GradType*>>* hist_buffers,
      std::vector<std::vector<uint8_t>>* hist_is_init);

    void BuildNodeStat(const GHistIndexMatrix &gmat,
                        DMatrix *p_fmat,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair_h,
                        int32_t nid);

    void BuildNodeStatBatch(
        const GHistIndexMatrix &gmat,
        DMatrix *p_fmat,
        RegTree *p_tree,
        const std::vector<GradientPair> &gpair_h,
        const std::vector<ExpandEntry>& nodes);

    int32_t FindSplitCond(int32_t nid,
                          RegTree *p_tree,
                          const GHistIndexMatrix &gmat);

    void CreateNewNodesBatch(
        const std::vector<ExpandEntry>& nodes,
        const GHistIndexMatrix &gmat,
        const ColumnMatrix &column_matrix,
        DMatrix *p_fmat,
        RegTree *p_tree,
        int *num_leaves,
        int depth,
        unsigned *timestamp,
        std::vector<ExpandEntry> *temp_qexpand_depth);

    template<typename TaskType, typename NodeType>
    void CreateTasksForApplySplit(
          const std::vector<ExpandEntry>& nodes,
          const GHistIndexMatrix &gmat,
          RegTree *p_tree,
          int *num_leaves,
          const int depth,
          const size_t block_size,
          std::vector<TaskType>* tasks,
          std::vector<NodeType>* nodes_bounds);

    void CreateTasksForBuildHist(
        size_t block_size_rows,
        size_t nthread,
        const std::vector<ExpandEntry>& nodes,
        std::vector<std::vector<common::GradStatHist::GradType*>>* hist_buffers,
        std::vector<std::vector<uint8_t>>* hist_is_init,
        std::vector<std::vector<common::GradStatHist>>* grad_stats,
        std::vector<int32_t>* task_nid,
        std::vector<int32_t>* task_node_idx,
        std::vector<int32_t>* task_block_idx);

    inline static bool LossGuide(ExpandEntry lhs, ExpandEntry rhs) {
      if (lhs.loss_chg == rhs.loss_chg) {
        return lhs.timestamp > rhs.timestamp;  // favor small timestamp
      } else {
        return lhs.loss_chg < rhs.loss_chg;  // favor large loss_chg
      }
    }

    HistCollection hist_buff_;

    //  --data fields--
    const TrainParam& param_;
    // number of omp thread used during training
    int nthread_;
    common::ColumnSampler column_sampler_;
    std::vector<SplitEntry> best_split_tloc_;
    std::vector<size_t> buffer_for_partition_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode_;
    /*! \brief culmulative histogram of gradients. */
    HistCollection hist_;
    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;
    /*! \brief local prediction cache; maps node id to leaf value */
    std::vector<float> leaf_value_cache_;

    GHistBuilder hist_builder_;
    std::unique_ptr<TreeUpdater> pruner_;
    std::unique_ptr<SplitEvaluator> spliteval_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_;
    const DMatrix* p_last_fmat_;
    LeaveIndexCache* index_cache_;

    using ExpandQueue =
       std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                           std::function<bool(ExpandEntry, ExpandEntry)>>;

    std::unique_ptr<ExpandQueue> qexpand_loss_guided_;
    std::vector<ExpandEntry> qexpand_depth_wise_;
    // key is the node id which should be calculated by Subtraction Trick, value is the node which
    // provides the evidence for substracts
    std::unordered_map<int, int> nodes_for_subtraction_trick_;

    enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;

    common::Monitor perf_monitor;
    rabit::Reducer<common::GradStatHist, common::GradStatHist::Reduce> histred_;
  };

  std::unique_ptr<Builder> builder_;
  std::unique_ptr<TreeUpdater> pruner_;
  std::unique_ptr<SplitEvaluator> spliteval_;
};

}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
