#include <future>
#include <vector>

#include "xgboost/tree_updater.h"
#include "xgboost/base.h"
#include "xgboost/json.h"
#include "hist/row_partitioner.h"
#include "hist/evaluate_splits.h"
#include "hist/param.h"
#include "constraints.h"
#include "driver.h"
#include "param.h"
#include "../common/random.h"
#include "updater_global_approx.h"

namespace xgboost {
namespace tree {

template <typename GradientSumT> class GloablApproxBuilder {
 protected:
  TrainParam param_;
  ApproxEvaluator<GradientSumT> evaluator_;
  ApproxHistogramBuilder<GradientSumT> histogram_builder_;

  ApproxRowPartitioner partitioner_;
  RegTree* p_last_tree_ {nullptr};
  common::Monitor* monitor_;

 public:
  void InitData(MetaInfo const& info, common::GHistIndexMatrix const &index) {
    monitor_->Start(__func__);
    partitioner_ = ApproxRowPartitioner(info.num_row_);
    histogram_builder_ = ApproxHistogramBuilder<GradientSumT>(index.cut.TotalBins());
    monitor_->Stop(__func__);
  }

  LocalExpandEntry InitRoot(common::GHistIndexMatrix const &m,
                            std::vector<GradientPair> const &gpair,
                            RegTree *p_tree) {
    LocalExpandEntry best;
    best.nid = RegTree::kRoot;
    best.depth = 0;
    GradStats root_sum;
    monitor_->Start("Root::SumGradient");
    for (auto const& g : gpair) {
      root_sum.Add(g);
    }
    monitor_->Stop("Root::SumGradient");

    rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double *>(&root_sum), 2);
    monitor_->Start("Root::BuildNodeHistogram");
    histogram_builder_.BuildNodeHistogram(gpair, partitioner_.Partitions(), m,
                                          m.IsDense(), {RegTree::kRoot});
    monitor_->Stop("Root::BuildNodeHistogram");

    auto weight = evaluator_.InitRoot(root_sum);
    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    auto const& histograms = histogram_builder_.Histograms();
    monitor_->Start("Root::EvaluateSplits");
    evaluator_.EvaluateSplits(histograms, m, *p_tree, {&best});
    monitor_->Stop("Root::EvaluateSplits");

    return best;
  }

  void UpdatePredictionCache(const DMatrix *data, VectorView<float> out_preds) {
    monitor_->Start(__func__);
    // Caching prediction seems redundant for approx tree method, as sketching
    // takes up majority of training time.
    CHECK(p_last_tree_);

    size_t n_nodes = p_last_tree_->GetNodes().size();
    CHECK_EQ(partitioner_.Size(), n_nodes);
    common::BlockedSpace2d space(
        n_nodes, [&](size_t node) { return partitioner_[node].Size(); }, 1024);

    auto evaluator = evaluator_.GetEvaluator();
    auto const &tree = *p_last_tree_;
    auto const &snode = evaluator_.Stats();
    common::ParallelFor2d(
        space, omp_get_max_threads(), [&](size_t nidx, common::Range1d r) {
          if (tree[nidx].IsLeaf()) {
            const auto rowset = partitioner_[nidx];
            auto const &stats = snode.at(nidx);
            auto leaf_value =
                evaluator.CalcWeight(nidx, param_, GradStats{stats.stats}) *
                param_.learning_rate;
            for (const size_t *it = rowset.begin + r.begin();
                 it < rowset.begin + r.end(); ++it) {
              out_preds[*it] += leaf_value;
            }
          }
        });
    monitor_->Stop(__func__);
  }

 public:
  explicit GloablApproxBuilder(TrainParam param, MetaInfo const &info,
                               common::Monitor *monitor)
      : param_{std::move(param)}, evaluator_{param_, info}, monitor_{monitor} {}

  void UpdateTree(RegTree *p_tree, MetaInfo const& info,
                  std::vector<GradientPair> const &gpair,
                  common::GHistIndexMatrix const &index) {
    p_last_tree_ = p_tree;
    this->InitData(info, index);
    UpdateTreeWithDriver(
        param_, p_tree, gpair,
        [&]() { return this->InitRoot(index, gpair, p_tree); },
        [&](LocalExpandEntry const &candidate) {
          evaluator_.ApplyTreeSplit(candidate, param_, p_tree);
        },
        [&](std::vector<LocalExpandEntry> const &applied) {
          partitioner_.UpdatePosition(index, applied, p_tree);
        },
        [&](std::vector<LocalExpandEntry> const &valid_candidates) {
          histogram_builder_.BuildHistogram(gpair, valid_candidates,
                                            partitioner_.Partitions(),
                                            index.IsDense(), index, p_tree);
        },
        [&](std::vector<LocalExpandEntry *> &best_splits) {
          auto const &histograms = histogram_builder_.Histograms();
          evaluator_.EvaluateSplits(histograms, index, *p_tree, best_splits);
        },
        monitor_);
  }
};

class GlobalApproxUpdater : public TreeUpdater {
  TrainParam param_;
  common::Monitor monitor_;
  CPUHistMakerTrainParam hist_param_;
  std::vector<bst_row_t> columns_size_;

  std::unique_ptr<GloablApproxBuilder<float>> f32_impl_;
  std::unique_ptr<GloablApproxBuilder<double>> f64_impl_;
  DMatrix* cached_ { nullptr };

 public:
  GlobalApproxUpdater() {
    monitor_.Init(__func__);
  }

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

  char const *Name() const override { return "grow_global_approx_histmaker"; }

  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
              const std::vector<RegTree *> &trees) override {
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();

    if (hist_param_.single_precision_histogram) {
      f32_impl_ = std::make_unique<GloablApproxBuilder<float>>(
          param_, m->Info(), &monitor_);
    } else {
      f64_impl_ = std::make_unique<GloablApproxBuilder<double>>(
          param_, m->Info(), &monitor_);
    }

    CHECK(!param_.enable_feature_grouping)
        << "Feature grouping is not implemented for approx.";

    std::vector<GradientPair> h_gpair;
    ApproxLazyInitData(param_, gpair, m, cached_, &h_gpair, &columns_size_);
    cached_ = m;
    auto const &info = m->Info();

    common::HistogramCuts cuts;
    std::vector<float> hessians(h_gpair.size());
    std::transform(h_gpair.cbegin(), h_gpair.cend(), hessians.begin(),
                   [](auto const &g) { return g.GetHess(); });
    common::HostSketchContainer container(columns_size_, param_.max_bin, false);
    if (m->IsDense()) {
      // This is actually twice slower, but removes a copy of data.
      monitor_.Start("Dense Sketch");
      for (auto const& page : m->GetBatches<SparsePage>()) {
        container.PushRowPage(page, info);
      }
      monitor_.Stop("Dense Sketch");
    } else {
      m->GetBatches<SortedCSCPage>();
      monitor_.Start("Sparse Sketch");
      for (auto const &page : m->GetBatches<SortedCSCPage>()) {
        container.PushSortedCSC(page, info, hessians);
      }
      monitor_.Stop("Sparse Sketch");
    }
    container.MakeCuts(&cuts);

    monitor_.Start("GHistIndexMatrix");
    common::GHistIndexMatrix gidx;
    gidx.Init(m, cuts, param_.max_bin);
    monitor_.Stop("GHistIndexMatrix");

    for (auto& p_tree : trees) {
      if (hist_param_.single_precision_histogram) {
        this->f32_impl_->UpdateTree(p_tree, info, h_gpair, gidx);
      } else {
        this->f64_impl_->UpdateTree(p_tree, info, h_gpair, gidx);
      }
    }
    param_.learning_rate = lr;
  }

  bool
  UpdatePredictionCache(const DMatrix *data,
                        VectorView<float> out_preds) override {
    if (data != cached_) { return false; }

    if (hist_param_.single_precision_histogram) {
      this->f32_impl_->UpdatePredictionCache(data, out_preds);
    } else {
      this->f64_impl_->UpdatePredictionCache(data, out_preds);
    }
    return true;
  }
};

#if !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(grow_global_approx_histmaker);
XGBOOST_REGISTER_TREE_UPDATER(GlobalHistMaker, "grow_global_approx_histmaker")
    .describe("Tree constructor that uses approximate histogram construction "
              "for each node.")
    .set_body([]() { return new GlobalApproxUpdater(); });
#endif  // !defined(GTEST_TEST)
}  // namespace tree
}  // namespace xgboost
