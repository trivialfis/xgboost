#include "xgboost/tree_updater.h"
#include "xgboost/base.h"
#include "xgboost/json.h"
#include "hist/row_partitioner.h"
#include "hist/evaluate_splits.h"
#include "updater_global_approx.h"
#include "updater_hist.h"

namespace xgboost {
namespace tree {

template <typename GradientSumT> class HistBuilder {
 protected:
  TrainParam param_;
  ApproxEvaluator<GradientSumT> evaluator_;
  ApproxHistogramBuilder<GradientSumT> histogram_builder_;

  HistRowPartitioner partitioner_;
  RegTree* p_last_tree_ {nullptr};
  common::Monitor* monitor_;

 public:
  void InitData(MetaInfo const& info, common::GHistIndexMatrix const &index) {
    monitor_->Start(__func__);
    partitioner_ = HistRowPartitioner(info.num_row_);
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
    for (auto const& g : gpair) {
      root_sum.Add(g);
    }
    rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double *>(&root_sum), 2);
    histogram_builder_.BuildNodeHistogram(gpair, partitioner_.Partitions(), m,
                                          m.IsDense(), {RegTree::kRoot});
    auto weight = evaluator_.InitRoot(root_sum);
    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    auto const& histograms = histogram_builder_.Histograms();
    evaluator_.EvaluateSplits(histograms, m, *p_tree, {&best});

    return best;
  }

  void UpdatePredictionCache(const DMatrix *data,
                             VectorView<bst_float> out_preds) {
    monitor_->Start(__func__);
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
  explicit HistBuilder(TrainParam param, MetaInfo const &info,
                               common::Monitor *monitor)
      : param_{std::move(param)}, evaluator_{param_, info}, monitor_{monitor} {}

  void UpdateTree(RegTree *p_tree, MetaInfo const& info,
                  std::vector<GradientPair> const &gpair,
                  common::GHistIndexMatrix const &index,
                  common::ColumnMatrix const& columns) {
    p_last_tree_ = p_tree;
    this->InitData(info, index);

    std::vector<GradientPair> sampled;
    auto const &h_gpair = gpair;
    sampled.resize(h_gpair.size());
    std::copy(h_gpair.cbegin(), h_gpair.cend(), sampled.begin());
    auto &rnd = common::GlobalRandom();
    if (param_.subsample != 1.0) {
      CHECK(param_.sampling_method != TrainParam::kGradientBased)
          << "Gradient based sampling is not supported for approx tree method.";
      std::bernoulli_distribution coin_flip(param_.subsample);
      std::transform(sampled.begin(), sampled.end(), sampled.begin(),
                     [&](GradientPair &g) {
                       if (coin_flip(rnd)) {
                         return g;
                       } else {
                         return GradientPair{};
                       }
                     });
    }

    UpdateTreeWithDriver(
        param_, p_tree, sampled,
        [&]() { return this->InitRoot(index, sampled, p_tree); },
        [&](LocalExpandEntry const &candidate) {
          evaluator_.ApplyTreeSplit(candidate, param_, p_tree);
        },
        [&](std::vector<LocalExpandEntry> const &applied) {
          partitioner_.UpdatePosition(index, columns, applied, p_tree);
        },
        [&](std::vector<LocalExpandEntry> const &valid_candidates) {
          histogram_builder_.BuildHistogram(sampled, valid_candidates,
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

class HistUpdater : public TreeUpdater {
  TrainParam param_;
  common::Monitor monitor_;
  CPUHistMakerTrainParam hist_param_;
  std::unique_ptr<HistBuilder<float>> f32_impl_;
  std::unique_ptr<HistBuilder<double>> f64_impl_;
  DMatrix* cached_ { nullptr };
  common::GHistIndexMatrix gidx_;
  common::ColumnMatrix columns_;

 public:
  HistUpdater() {
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

  char const *Name() const override { return "grow_fast_histmaker"; }

  void InitData(DMatrix* m) {
    if (m != cached_) {
      cached_ = m;
      gidx_.Init(m, param_.max_bin);
      columns_.Init(gidx_, param_.sparse_threshold);
    }
  }

  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
              const std::vector<RegTree *> &trees) override {
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();

    if (hist_param_.single_precision_histogram) {
      f32_impl_ =
          std::make_unique<HistBuilder<float>>(param_, m->Info(), &monitor_);
    } else {
      f64_impl_ =
          std::make_unique<HistBuilder<double>>(param_, m->Info(), &monitor_);
    }

    InitData(m);
    auto const &info = m->Info();
    auto const& h_gpair = gpair->ConstHostVector();

    for (auto p_tree : trees) {
      if (hist_param_.single_precision_histogram) {
        this->f32_impl_->UpdateTree(p_tree, info, h_gpair, gidx_, columns_);
      } else {
        this->f64_impl_->UpdateTree(p_tree, info, h_gpair, gidx_, columns_);
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

XGBOOST_REGISTER_TREE_UPDATER(HistUpdater, "grow_fast_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body(
    []() {
      return new HistUpdater();
    });
}  // namespace tree
}  // namespace xgboost
