/*!
 * Copyright 2014-2019 by Contributors
 * \file tree_updater.h
 * \brief General primitive for tree learning,
 *   Updating a collection of trees given the information.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_H_
#define XGBOOST_TREE_UPDATER_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/tree_model.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>

#include <functional>
#include <vector>
#include <utility>
#include <string>

namespace xgboost {
/*!
 * \brief interface of tree update module, that performs update of a tree.
 */
class TreeUpdater {
 public:
  /*! \brief virtual destructor */
  virtual ~TreeUpdater() = default;
  /*!
   * \brief Initialize the updater with given arguments.
   * \param args arguments to the objective function.
   */
  virtual void Configure(const Args& args) = 0;
  /*!
   * \brief perform update to the tree models
   * \param gpair the gradient pair statistics of the data
   * \param data The data matrix passed to the updater.
   * \param trees references the trees to be updated, updater will change the content of trees
   *   note: all the trees in the vector are updated, with the same statistics,
   *         but maybe different random seeds, usually one tree is passed in at a time,
   *         there can be multiple trees when we train random forest style model
   */
  virtual void Update(HostDeviceVector<GradientPair>* gpair,
                      DMatrix* data,
                      const std::vector<RegTree*>& trees) = 0;

  /*!
   * \brief determines whether updater has enough knowledge about a given dataset
   *        to quickly update prediction cache its training data and performs the
   *        update if possible.
   * \param data: data matrix
   * \param out_preds: prediction cache to be updated
   * \return boolean indicating whether updater has capability to update
   *         the prediction cache. If true, the prediction cache will have been
   *         updated by the time this function returns.
   */
  virtual bool UpdatePredictionCache(const DMatrix* data,
                                     HostDeviceVector<bst_float>* out_preds) {
    return false;
  }

  virtual char const* Name() const = 0;

  struct LeaveIndexCache;

  /*!
   * \brief Create a tree updater given name
   * \param name Name of the tree updater.
   */
  static TreeUpdater* Create(const std::string& name, LeaveIndexCache* cache,
                             GenericParameter const* tparam);

  /** \brief Used to demarcate a contiguous set of row indices associated with
   * some tree node. */
  struct Segment {
    size_t begin;
    size_t end;

    Segment(size_t begin, size_t end) : begin(begin), end(end) {
      CHECK_GE(end, begin);
    }
    Segment() = default;
    size_t Size() const { return end - begin; }
  };

  struct LeaveIndexCache {
    HostDeviceVector<size_t> row_index;
    std::vector<Segment> ridx_segments;

   public:
    using const_iterator = typename std::vector<Segment>::const_iterator;
    using iterator = typename std::vector<Segment>::iterator;
    using value_type = Segment;
    using pointer = value_type*;
    using reference = value_type&;
    using index_type = int32_t;  // Node ID type

    LeaveIndexCache() = default;

    std::vector<Segment>::const_iterator cbegin() const {
      return ridx_segments.cbegin();
    }
    std::vector<Segment>::const_iterator cend() const {
      return ridx_segments.cend();
    }

    /*! \brief return corresponding element set given the node_id */
    common::Span<size_t const> operator[](unsigned node_id) const;
    common::Span<size_t> operator[](unsigned node_id);

    void Init() {
      CHECK_EQ(ridx_segments.size(), 0U);
      // FIXME(trivialfis): https://github.com/dmlc/xgboost/issues/2800
      const size_t begin = 0;
      const size_t end = row_index.Size();
      CHECK_NE(row_index.Size(), 0);
      ridx_segments.emplace_back(begin, end);
    }

    common::Span<size_t> GetRows() {
      auto* ptr = row_index.HostVector().data();
      auto size = row_index.HostVector().size();
      return common::Span<size_t>{ ptr, size };
    }
    std::vector<size_t>& HostRowIndices() {
      return this->row_index.HostVector();
    }

    void AddSplit(unsigned node_id, size_t iLeft, unsigned left_node_id,
                  unsigned right_node_id);

    void Clear() {
      this->row_index.Resize(0);
      this->ridx_segments.clear();
    }
  };

 protected:
  void SetCache(LeaveIndexCache* cache) {
    this->index_cache_ = cache;
  }

 protected:
  GenericParameter const* tparam_ {nullptr};
  LeaveIndexCache* index_cache_ {nullptr};
};

/*!
 * \brief Registry entry for tree updater.
 */
struct TreeUpdaterReg
    : public dmlc::FunctionRegEntryBase<TreeUpdaterReg,
                                        std::function<TreeUpdater* ()> > {
};

/*!
 * \brief Macro to register tree updater.
 *
 * \code
 * // example of registering a objective ndcg@k
 * XGBOOST_REGISTER_TREE_UPDATER(ColMaker, "colmaker")
 * .describe("Column based tree maker.")
 * .set_body([]() {
 *     return new ColMaker<TStats>();
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_TREE_UPDATER(UniqueId, Name)                   \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::TreeUpdaterReg&               \
  __make_ ## TreeUpdaterReg ## _ ## UniqueId ## __ =                    \
      ::dmlc::Registry< ::xgboost::TreeUpdaterReg>::Get()->__REGISTER__(Name)

}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_H_
