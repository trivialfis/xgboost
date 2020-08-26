#include "xgboost/tree_updater.h"
#include "xgboost/json.h"
#include "hist/param.h"
#include "constraints.h"
#include "driver.h"
#include "../common/random.h"
#include "hist/evaluate_splits.h"
#include "hist/row_partitioner.h"
#include "hist/evaluate_splits.h"
#include "param.h"
#include "../common/quantile.h"
#include "updater_approx.h"

namespace xgboost {
namespace tree {
template <typename GradientSumT> class LocalApproxBuilder {
 protected:
  TrainParam param_;
  std::vector<bst_row_t> columns_size_;
  common::GHistIndexMatrix index_;
  ApproxHistogramBuilder<GradientSumT> histogram_builder_;
  ApproxEvaluator<GradientSumT> evaluator_;
  ApproxRowPartitioner partitioner_;

  RegTree* p_last_tree_ {nullptr};
  common::Monitor* monitor_;

  void InitData(DMatrix *m, std::vector<GradientPair> const &gpair) {
    monitor_->Start(__func__);
    auto const &info = m->Info();
    auto sketches =
        common::HostSketchContainer(columns_size_, param_.max_bin, false);
    for (auto const& page : m->GetBatches<SparsePage>()) {
      sketches.PushRowPage(page, info);
    }
    common::HistogramCuts cuts;
    sketches.MakeCuts(&cuts);
    index_.Init(m, cuts, param_.max_bin);

    partitioner_ = ApproxRowPartitioner(info.num_row_);
    histogram_builder_ = ApproxHistogramBuilder<GradientSumT>(index_.cut.TotalBins());
    monitor_->Stop(__func__);
  }

  LocalExpandEntry InitRoot(DMatrix* m, std::vector<GradientPair> const &gpair,
                            RegTree *p_tree) {
    monitor_->Start(__func__);
    LocalExpandEntry best{RegTree::kRoot, 0, {}};
    GradStats root_sum;
    for (auto const& g : gpair) {
      root_sum.Add(g);
    }
    rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double *>(&root_sum), 2);

    histogram_builder_.BuildNodeHistogram(gpair, partitioner_.Partitions(), index_,
                                          index_.IsDense(), {RegTree::kRoot});
    auto weight = evaluator_.InitRoot(root_sum);
    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    auto const& histograms = histogram_builder_.Histograms();
    evaluator_.EvaluateSplits(histograms, index_, *p_tree, {&best});

    monitor_->Stop(__func__);
    return best;
  }

  void UpdateSketch(DMatrix *m, std::vector<GradientPair> const &gpair, bst_node_t nidx) {
    auto row_set = partitioner_[nidx];
    auto sketches =
        common::HostSketchContainer(columns_size_, param_.max_bin, false);
    for (auto const& page : m->GetBatches<SparsePage>()) {
      // FIXME: External memory
      for (size_t i = 0; i < row_set.Size(); ++i) {
        size_t ridx = *(row_set.begin + i);
        auto inst = page[ridx];
        sketches.PushInstance(inst, gpair[ridx].GetHess());
      }
      common::HistogramCuts cuts;
      sketches.MakeCuts(&cuts);
      auto row_set_span = common::Span<size_t const>{row_set.begin, row_set.Size()};
      index_.SetIndexForRowSet(page, row_set_span, cuts);
    }
  }

 public:
  LocalApproxBuilder(TrainParam param, MetaInfo const &info,
                     std::vector<bst_row_t> columns_size,
                     common::Monitor *monitor)
      : param_{param}, columns_size_{std::move(columns_size)},
        evaluator_(param, info), monitor_{monitor} {}

  void UpdateTree(RegTree *p_tree, DMatrix *m,
                  std::vector<GradientPair> const &gpair) {
    p_last_tree_ = p_tree;
    this->InitData(m, gpair);
    UpdateTreeWithDriver(
        param_, p_tree, gpair,
        [&]() { return this->InitRoot(m, gpair, p_tree); },
        [&](LocalExpandEntry const &candidate) {
          evaluator_.ApplyTreeSplit(candidate, param_, p_tree);
        },
        [&](std::vector<LocalExpandEntry> const &applied) {
          partitioner_.UpdatePosition(index_, applied, p_tree);
        },
        [&](std::vector<LocalExpandEntry> const &valid_candidates) {
          for (auto candidate : valid_candidates) {
            this->UpdateSketch(m, gpair, candidate.nid);
          }
          histogram_builder_.BuildHistogram(gpair, valid_candidates,
                                            partitioner_.Partitions(),
                                            index_.IsDense(), index_, p_tree);
        },
        [&](std::vector<LocalExpandEntry *> &best_splits) {
          auto const &histograms = histogram_builder_.Histograms();
          evaluator_.EvaluateSplits(histograms, index_, *p_tree, best_splits);
        });
  }
};

class LocalApproxUpdater : public TreeUpdater {
  std::unique_ptr<LocalApproxBuilder<float>> f32_impl_;
  std::unique_ptr<LocalApproxBuilder<double>> f64_impl_;
  std::vector<bst_row_t> columns_size_;
  common::Monitor monitor_;

  TrainParam param_;
  CPUHistMakerTrainParam hist_param_;
  DMatrix* cached_ { nullptr };

 public:
  void Configure(const Args& args) override {
    param_.UpdateAllowUnknown(args);
    hist_param_.UpdateAllowUnknown(args);
  }
  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    FromJson(config.at("hist_param"), &this->hist_param_);
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["hist_param"] = ToJson(hist_param_);
  }

  char const *Name() const override { return "grow_local_approx_histmaker"; }

  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
              const std::vector<RegTree *> &trees) override {
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();

    std::vector<GradientPair> h_gpair;
    ApproxLazyInitData(param_, gpair, m, cached_, &h_gpair, &columns_size_);
    cached_ = m;

    if (hist_param_.single_precision_histogram) {
      f32_impl_ = std::make_unique<LocalApproxBuilder<float>>(
          param_, m->Info(), columns_size_,
          &monitor_);
    } else {
      f64_impl_ = std::make_unique<LocalApproxBuilder<double>>(
          param_, m->Info(), columns_size_,
          &monitor_);
    }

    CHECK(!param_.enable_feature_grouping)
        << "Feature grouping is not implemented for approx.";

    for (auto p_tree : trees) {
      if (hist_param_.single_precision_histogram) {
        this->f32_impl_->UpdateTree(p_tree, m, h_gpair);
      } else {
        this->f64_impl_->UpdateTree(p_tree, m, h_gpair);
      }
    }
    param_.learning_rate = lr;
  }
};

#if !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(grow_local_approx_histmaker);
XGBOOST_REGISTER_TREE_UPDATER(LocalHistMaker, "grow_local_approx_histmaker")
    .describe("Tree constructor that uses approximate histogram construction "
              "for each node.")
    .set_body([]() { return new LocalApproxUpdater(); });
#endif  // !defined(GTEST_TEST)
}  // namespace tree
}  // namespace xgboost
