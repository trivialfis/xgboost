/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <limits>
#include <utility>
#include "quantile.h"
#include "hist_util.h"

namespace xgboost {
namespace common {

HostSketchContainer::HostSketchContainer(std::vector<bst_row_t> columns_size,
                                         int32_t max_bins, bool use_group)
    : columns_size_{std::move(columns_size)}, max_bins_{max_bins},
      use_group_ind_{use_group} {
  CHECK_NE(columns_size_.size(), 0);
  sketches_.resize(columns_size_.size());
  for (size_t i = 0; i < sketches_.size(); ++i) {
    auto n_bins = std::min(static_cast<size_t>(max_bins_), columns_size_[i]);
    n_bins = std::max(n_bins, 1ul);
    auto eps = 1.0 / (static_cast<float>(n_bins) * WQSketch::kFactor);
    sketches_[i].Init(columns_size_[i], eps);
  }
}

void HostSketchContainer::PushRowPage(SparsePage const &page,
                                      MetaInfo const &info) {
  int nthread = omp_get_max_threads();
  CHECK_EQ(sketches_.size(), info.num_col_);

  // Data groups, used in ranking.
  std::vector<bst_uint> const &group_ptr = info.group_ptr_;
  size_t const num_groups = group_ptr.size() == 0 ? 0 : group_ptr.size() - 1;
  // Use group index for weights?
  size_t group_ind = 0;
  auto batch = page.GetView();
  if (use_group_ind_) {
    group_ind = this->SearchGroupIndFromRow(group_ptr, page.base_rowid);
  }
  dmlc::OMPException exec;
  // Parallel over columns.  Asumming the data is dense, each thread owns a set of
  // consecutive columns.
  unsigned const nstep =
      static_cast<unsigned>((info.num_col_ + nthread - 1) / nthread);
  unsigned const ncol = static_cast<unsigned>(info.num_col_);
  auto is_dense = info.num_nonzero_ == info.num_col_ * info.num_row_;
#pragma omp parallel num_threads(nthread) firstprivate(group_ind, use_group_ind_)
  {
    CHECK_EQ(nthread, omp_get_num_threads());
    auto tid = static_cast<unsigned>(omp_get_thread_num());
    unsigned begin = std::min(nstep * tid, ncol);
    unsigned end = std::min(nstep * (tid + 1), ncol);

    // do not iterate if no columns are assigned to the thread
    if (begin < end && end <= ncol) {
      for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
        size_t const ridx = page.base_rowid + i;
        SparsePage::Inst const inst = batch[i];
        if (use_group_ind_ && group_ptr[group_ind] == ridx &&
            // maximum equals to weights.size() - 1
            group_ind < num_groups - 1) {
          // move to next group
          group_ind++;
        }
        size_t w_idx = use_group_ind_ ? group_ind : ridx;
        auto w = info.GetWeight(w_idx);
        if (is_dense) {
          auto data = inst.data();
          for (size_t ii = begin; ii < end; ii++) {
            sketches_[ii].Push(data[ii].fvalue, w);
          }
        } else {
          auto p_data = inst.data();
          auto p_end = inst.data() + inst.size();
          for (auto it = p_data;
               it->index < end && it != p_end; ++it) {
            if (it->index >= begin) {
              sketches_[it->index].Push(it->fvalue, w);
            }
          }
        }
      }
    }
  }
  exec.Rethrow();
}

void AddCutPoint(WQuantileSketch<float, float>::SummaryContainer const &summary,
                 int max_bin, HistogramCuts *cuts) {
  size_t required_cuts = std::min(summary.size, static_cast<size_t>(max_bin));
  auto& cut_values = cuts->cut_values_.HostVector();
  for (size_t i = 1; i < required_cuts; ++i) {
    bst_float cpt = summary.data[i].value;
    if (i == 1 || cpt > cuts->cut_values_.ConstHostVector().back()) {
      cut_values.push_back(cpt);
    }
  }
}

void HostSketchContainer::MakeCuts(HistogramCuts* cuts) {
  rabit::Allreduce<rabit::op::Sum>(columns_size_.data(), columns_size_.size());
  std::vector<WQSketch::SummaryContainer> reduced(sketches_.size());
  std::vector<int32_t> num_cuts;
  size_t nbytes = 0;
  for (size_t i = 0; i < sketches_.size(); ++i) {
    size_t intermediate_num_cuts = std::min(
        columns_size_[i], static_cast<size_t>(max_bins_ * WQSketch::kFactor));
    WQSketch::SummaryContainer out;
    sketches_[i].GetSummary(&out);
    reduced[i].Reserve(intermediate_num_cuts);
    reduced[i].SetPrune(out, intermediate_num_cuts);
    num_cuts.push_back(intermediate_num_cuts);
    nbytes = std::max(
        WQSketch::SummaryContainer::CalcMemCost(intermediate_num_cuts), nbytes);
  }

  if (rabit::IsDistributed()) {
    // FIXME(trivialfis): This call will allocate nbytes * num_columns on rabit, which
    // may generate oom error when data is sparse.  To fix it, we need to:
    //   - gather the column offsets over all workers.
    //   - run rabit::allgather on sketch data to collect all data.
    //   - merge all gathered sketches based on worker offsets and column offsets of data
    //     from each worker.
    // See GPU implementation for details.
    rabit::SerializeReducer<WQSketch::SummaryContainer> sreducer;
    sreducer.Allreduce(dmlc::BeginPtr(reduced), nbytes, reduced.size());
  }

  cuts->min_vals_.HostVector().resize(sketches_.size());
  for (size_t fid = 0; fid < reduced.size(); ++fid) {
    WQSketch::SummaryContainer a;
    size_t max_num_bins = std::min(num_cuts[fid], max_bins_);
    a.Reserve(max_num_bins + 1);
    a.SetPrune(reduced[fid], max_num_bins + 1);
    const bst_float mval = a.data[0].value;
    cuts->min_vals_.HostVector()[fid] = mval - (fabs(mval) + 1e-5);
    AddCutPoint(a, max_num_bins, cuts);
    // push a value that is greater than anything
    const bst_float cpt
      = (a.size > 0) ? a.data[a.size - 1].value : cuts->min_vals_.HostVector()[fid];
    // this must be bigger than last value in a scale
    const bst_float last = cpt + (fabs(cpt) + 1e-5);
    cuts->cut_values_.HostVector().push_back(last);

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(cuts->cut_values_.HostVector().size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(cuts->cut_values_.HostVector().size());
    CHECK_GT(cut_size, cuts->cut_ptrs_.HostVector().back());
    cuts->cut_ptrs_.HostVector().push_back(cut_size);
  }
}
}  // namespace common
}  // namespace xgboost
