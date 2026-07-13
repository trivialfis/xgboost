/**
 * Copyright 2021-2023, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
#define XGBOOST_TREE_HIST_EXPAND_ENTRY_H_

#include <algorithm>    // for all_of
#include <ostream>      // for ostream
#include <string>       // for string
#include <type_traits>  // for add_const_t
#include <utility>      // for move
#include <vector>       // for vector

#include "../../common/type.h"  // for EraseType
#include "../param.h"           // for SplitEntry, SplitEntryContainer, TrainParam
#include "xgboost/base.h"       // for GradientPairPrecise, bst_node_t
#include "xgboost/json.h"       // for Json

namespace xgboost::tree {
/**
 * \brief Structure for storing tree split candidate.
 */
template <typename Impl>
struct ExpandEntryImpl {
  bst_node_t nid{0};
  bst_node_t depth{0};

  [[nodiscard]] float GetLossChange() const {
    return static_cast<Impl const*>(this)->split.loss_chg;
  }
  [[nodiscard]] bst_node_t GetNodeId() const { return nid; }

  [[nodiscard]] bool IsValid(TrainParam const& param, bst_node_t num_leaves) const {
    return static_cast<Impl const*>(this)->IsValidImpl(param, num_leaves);
  }
};

struct CPUExpandEntry : public ExpandEntryImpl<CPUExpandEntry> {
  SplitEntry split;

  CPUExpandEntry() = default;
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth, SplitEntry split)
      : ExpandEntryImpl{nidx, depth}, split(std::move(split)) {}
  CPUExpandEntry(bst_node_t nidx, bst_node_t depth) : ExpandEntryImpl{nidx, depth} {}

  void Save(Json* p_out) const {
    auto& out = *p_out;
    out["nid"] = Integer{this->nid};
    out["depth"] = Integer{this->depth};

    out["split"] = Object{};
    auto& j_split = out["split"];
    j_split["loss_chg"] = this->split.loss_chg;
    j_split["sindex"] = Integer{this->split.sindex};
    j_split["split_value"] = this->split.split_value;

    auto const& cat_bits = this->split.cat_bits;
    auto s_cat_bits = common::Span{cat_bits.data(), cat_bits.size()};
    j_split["cat_bits"] = U8Array{s_cat_bits.size_bytes()};
    auto& j_cat_bits = get<U8Array>(j_split["cat_bits"]);
    using T = typename decltype(this->split.cat_bits)::value_type;
    auto erased =
        common::EraseType<std::add_const_t<T>, std::add_const_t<std::uint8_t>>(s_cat_bits);
    for (std::size_t i = 0; i < erased.size(); ++i) {
      j_cat_bits[i] = erased[i];
    }

    j_split["is_cat"] = Boolean{this->split.is_cat};
    this->SaveGrad(&j_split);
  }

  void Load(Json const& in) {
    this->nid = get<Integer const>(in["nid"]);
    this->depth = get<Integer const>(in["depth"]);

    auto const& j_split = in["split"];
    this->split.loss_chg = get<Number const>(j_split["loss_chg"]);
    this->split.sindex = get<Integer const>(j_split["sindex"]);
    this->split.split_value = get<Number const>(j_split["split_value"]);

    auto const& j_cat_bits = get<U8Array const>(j_split["cat_bits"]);
    using T = typename decltype(this->split.cat_bits)::value_type;
    auto restored = common::RestoreType<std::add_const_t<T>>(
        common::Span{j_cat_bits.data(), j_cat_bits.size()});
    this->split.cat_bits.resize(restored.size());
    for (std::size_t i = 0; i < restored.size(); ++i) {
      this->split.cat_bits[i] = restored[i];
    }

    this->split.is_cat = get<Boolean const>(j_split["is_cat"]);
    this->LoadGrad(j_split);
  }

  void SaveGrad(Json* p_out) const {
    auto& out = *p_out;
    auto save = [&](std::string const& name, GradStats const& sum) {
      out[name] = F64Array{2};
      auto& array = get<F64Array>(out[name]);
      array[0] = sum.GetGrad();
      array[1] = sum.GetHess();
    };
    save("left_sum", this->split.left_sum);
    save("right_sum", this->split.right_sum);
  }
  void LoadGrad(Json const& in) {
    auto const& left_sum = get<F64Array const>(in["left_sum"]);
    this->split.left_sum = GradStats{left_sum[0], left_sum[1]};
    auto const& right_sum = get<F64Array const>(in["right_sum"]);
    this->split.right_sum = GradStats{right_sum[0], right_sum[1]};
  }

  [[nodiscard]] bool IsValidImpl(TrainParam const& param, bst_node_t num_leaves) const {
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

  friend std::ostream& operator<<(std::ostream& os, CPUExpandEntry const& e) {
    os << "ExpandEntry:\n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "split:\n" << e.split << std::endl;
    return os;
  }
};

struct MultiExpandEntry : public ExpandEntryImpl<MultiExpandEntry> {
  SplitEntryContainer<std::vector<GradientPairPrecise>> split;

  MultiExpandEntry() = default;
  MultiExpandEntry(bst_node_t nidx, bst_node_t depth) : ExpandEntryImpl{nidx, depth} {}

  [[nodiscard]] bool IsValidImpl(TrainParam const& param, bst_node_t num_leaves) const {
    if (split.loss_chg <= kRtEps) return false;
    auto is_zero = [](auto const& sum) {
      return std::all_of(sum.cbegin(), sum.cend(),
                         [&](auto const& g) { return g.GetHess() - .0 == .0; });
    };
    if (is_zero(split.left_sum) || is_zero(split.right_sum)) {
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

  friend std::ostream& operator<<(std::ostream& os, MultiExpandEntry const& e) {
    os << "ExpandEntry: \n";
    os << "nidx: " << e.nid << "\n";
    os << "depth: " << e.depth << "\n";
    os << "loss: " << e.split.loss_chg << "\n";
    os << "split cond:" << e.split.split_value << "\n";
    os << "split ind:" << e.split.SplitIndex() << "\n";
    os << "left_sum: [";
    for (auto v : e.split.left_sum) {
      os << v << ", ";
    }
    os << "]\n";

    os << "right_sum: [";
    for (auto v : e.split.right_sum) {
      os << v << ", ";
    }
    os << "]\n";
    return os;
  }
};
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EXPAND_ENTRY_H_
