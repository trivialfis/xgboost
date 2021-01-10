#ifndef XGBOOST_TREE_DRIVER_H_
#define XGBOOST_TREE_DRIVER_H_
#include <functional>
#include <queue>
#include "param.h"

namespace xgboost {
namespace tree {
template <typename ExpandEntry>
inline bool DepthWise(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  return lhs.depth > rhs.depth;  // favor small depth
}

template <typename ExpandEntry>
inline bool LossGuide(const ExpandEntry& lhs, const ExpandEntry& rhs) {
  if (lhs.split.loss_chg == rhs.split.loss_chg) {
    return lhs.nid > rhs.nid;  // favor small timestamp
  } else {
    return lhs.split.loss_chg < rhs.split.loss_chg;  // favor large loss_chg
  }
}

template <typename ExpandEntry>
class DriverContainer {
  using ExpandQueue =
      std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                          std::function<bool(ExpandEntry, ExpandEntry)>>;

 public:
  explicit DriverContainer(TrainParam::TreeGrowPolicy policy)
      : policy_(policy),
        queue_(policy == TrainParam::kDepthWise ? DepthWise<ExpandEntry> : LossGuide<ExpandEntry>) {}
  template <typename EntryIterT>
  void Push(EntryIterT begin,EntryIterT end) {
    for (auto it = begin; it != end; ++it) {
      const ExpandEntry& e = *it;
      if (e.split.loss_chg > kRtEps) {
        queue_.push(e);
      }
    }
  }
  void Push(const std::vector<ExpandEntry> &entries) {
    this->Push(entries.begin(), entries.end());
  }
  // Return the set of nodes to be expanded
  // This set has no dependencies between entries so they may be expanded in
  // parallel or asynchronously
  std::vector<ExpandEntry> Pop() {
    if (queue_.empty()) return {};
    // Return a single entry for loss guided mode
    if (policy_ == TrainParam::kLossGuide) {
      ExpandEntry e = queue_.top();
      queue_.pop();
      return {e};
    }
    // Return nodes on same level for depth wise
    std::vector<ExpandEntry> result;
    ExpandEntry e = queue_.top();
    int level = e.depth;
    while (e.depth == level && !queue_.empty()) {
      queue_.pop();
      result.emplace_back(e);
      if (!queue_.empty()) {
        e = queue_.top();
      }
    }
    return result;
  }

 private:
  TrainParam::TreeGrowPolicy policy_;
  ExpandQueue queue_;
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_DRIVER_H_
