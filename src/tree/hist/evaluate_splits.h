#ifndef XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#define XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_

#include "xgboost/base.h"
#include "xgboost/tree_model.h"
#include "../param.h"
#include "../split_evaluator.h"
#include "../../common/hist_util.h"

namespace xgboost {
namespace tree {

struct LocalExpandEntry {
  int nid;
  int depth;
  SplitEntry split;
  LocalExpandEntry() = default;
  XGBOOST_DEVICE
  LocalExpandEntry(int nid, int depth, SplitEntry split)
      : nid(nid), depth(depth), split(std::move(split)) {}
  bool IsValid(const TrainParam& param, int num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    if (split.left_sum.GetHess() == 0 || split.right_sum.GetHess() == 0) {
      return false;
    }
    if (split.loss_chg < param.min_split_loss) {
      return false;
    }
    if (param.max_depth > 0 && depth == param.max_depth) {
      return false;
    }
    if (param.max_leaves > 0 && num_leaves == param.max_leaves) {
      return false;
    }
    return true;
  }

  static bool ChildIsValid(const TrainParam& param, int depth, int num_leaves) {
    if (param.max_depth > 0 && depth >= param.max_depth) return false;
    if (param.max_leaves > 0 && num_leaves >= param.max_leaves) return false;
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const LocalExpandEntry& e) {
    os << "ExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "left_sum: " << e.split.left_sum << "\n";
    os << "right_sum: " << e.split.right_sum << "\n";
    return os;
  }
};

template <typename GHistRowT, typename NodeEntry, typename SplitEntry, int d_step>
GradStats
EnumerateSplit(const common::GHistIndexMatrix &gmat, const GHistRowT &hist,
               const NodeEntry &snode, SplitEntry *p_best, bst_node_t nidx, bst_uint fid,
               TrainParam const& param, TreeEvaluator::SplitEvaluator<TrainParam> const& evaluator) {
  auto p_hist = hist.data();
  // aliases
  const std::vector<uint32_t> &cut_ptr = gmat.cut.Ptrs();
  const std::vector<bst_float> &cut_val = gmat.cut.Values();

  // statistics on both sides of split
  GradStats c;
  GradStats e;
  // best split so far
  SplitEntry best;

  // imin: index (offset) of the minimum value for feature fid
  //       need this for backward enumeration
  const auto imin = static_cast<int32_t>(cut_ptr[fid]);
  // ibegin, iend: smallest/largest cut points for feature fid
  // use int to allow for value -1
  int32_t ibegin, iend;
  if (d_step > 0) {
    ibegin = static_cast<int32_t>(cut_ptr[fid]);
    iend = static_cast<int32_t>(cut_ptr[fid + 1]);
  } else {
    ibegin = static_cast<int32_t>(cut_ptr[fid + 1]) - 1;
    iend = static_cast<int32_t>(cut_ptr[fid]) - 1;
  }

  for (int32_t i = ibegin; i != iend; i += d_step) {
    // start working
    // try to find a split
    e.Add(p_hist[i].GetGrad(), p_hist[i].GetHess());
    if (e.GetHess() >= param.min_child_weight) {
      c.SetSubstract(snode.stats, e);
      if (c.GetHess() >= param.min_child_weight) {
        bst_float loss_chg;
        bst_float split_pt;
        if (d_step > 0) {
          // forward enumeration: split at right bound of each bin
          loss_chg = static_cast<bst_float>(
              evaluator.CalcSplitGain(param, nidx, fid, e, c) -
              snode.root_gain);
          split_pt = cut_val[i];
          best.Update(loss_chg, fid, split_pt, d_step == -1, e, c);
        } else {
          // backward enumeration: split at left bound of each bin
          loss_chg = static_cast<bst_float>(
              evaluator.CalcSplitGain(param, nidx, fid, c, e) -
              snode.root_gain);
          if (i == imin) {
            // for leftmost bin, left bound is the smallest feature value
            split_pt = gmat.cut.MinValues()[fid];
          } else {
            split_pt = cut_val[i - 1];
          }
          best.Update(loss_chg, fid, split_pt, d_step == -1, c, e);
        }
      }
    }
  }
  p_best->Update(best);

  return e;
}

template <typename NodeEntry>
bool SplitContainsMissingValues(const GradStats e, const NodeEntry &snode) {
  if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
    return false;
  } else {
    return true;
  }
}

template <typename ExpandEntry>
void FindSplitConditions(const std::vector<ExpandEntry> &nodes,
                         const RegTree &tree, const common::GHistIndexMatrix &gmat,
                         std::vector<int32_t> *split_conditions) {
  const size_t n_nodes = nodes.size();
  split_conditions->resize(n_nodes);

  for (size_t i = 0; i < nodes.size(); ++i) {
    const int32_t nid = nodes[i].nid;
    const bst_uint fid = tree[nid].SplitIndex();
    const bst_float split_pt = tree[nid].SplitCond();
    const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
    const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
    int32_t split_cond = -1;
    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    CHECK_LT(upper_bound,
             static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    for (uint32_t i = lower_bound; i < upper_bound; ++i) {
      if (split_pt == gmat.cut.Values()[i]) {
        split_cond = static_cast<int32_t>(i);
      }
    }
    (*split_conditions)[i] = split_cond;
  }
}
};  // namespace tree
};  // namespace xgboost

#endif  // XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
