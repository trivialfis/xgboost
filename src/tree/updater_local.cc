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
  using WQSketch = common::WQuantileSketch<float, float>;

 protected:
  using GradientPairT =
      std::conditional_t<std::is_same<GradientSumT, float>::value, GradientPair,
                         GradientPairPrecise>;

  TrainParam param_;
  CPUHistMakerTrainParam hist_param_;

  TreeEvaluator evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  common::ColumnSampler column_sampler_;
  std::vector<int> monotonic_constraint_;

  ApproxHistogramBuilder<GradientSumT> histogram_builder_;

  common::HostSketchContainer sketches_;
  common::GHistIndexMatrix index_;

  ApproxRowPartitioner partitioner_;

  RegTree* p_last_tree_ {nullptr};
  common::Monitor* monitor_;

  std::vector<NodeEntry> snode_;

  void InitData(DMatrix *m, std::vector<GradientPair> const &gpair,
                common::GHistIndexMatrix const &index) {
    monitor_->Start(__func__);
    auto const &info = m->Info();

    partitioner_ = ApproxRowPartitioner(info.num_row_);

    column_sampler_.Init(info.num_col_, m->Info().feature_weigths.HostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, false);

    if (!param_.monotone_constraints.empty()) {
      monotonic_constraint_ = param_.monotone_constraints;
    } else {
      monotonic_constraint_.resize(info.num_col_, 0);
    }

    histogram_builder_ = ApproxHistogramBuilder<GradientSumT>(index.cut.TotalBins());
    monitor_->Stop(__func__);
  }

  LocalExpandEntry InitRoot(common::GHistIndexMatrix const &m,
                            std::vector<GradientPair> const &gpair,
                            RegTree *p_tree) {
    monitor_->Start(__func__);
    LocalExpandEntry best;
    best.nid = RegTree::kRoot;
    best.depth = 0;
    GradStats root_sum;
    for (auto const& g : gpair) {
      root_sum.Add(g);
    }
    rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double *>(&root_sum), 2);

    this->BuildNodeHistogram(gpair, m, m.IsDense(), {RegTree::kRoot});

    snode_.resize(p_tree->GetNodes().size());
    CHECK_EQ(snode_.size(), 1);
    auto root_evaluator = evaluator_.GetEvaluator();

    snode_[0].stats = GradStats{root_sum.GetGrad(), root_sum.GetHess()};
    snode_[0].root_gain = root_evaluator.CalcGain(RegTree::kRoot, param_,
                                                  GradStats{snode_[0].stats});
    auto weight = root_evaluator.CalcWeight(RegTree::kRoot, param_,
                                            GradStats{snode_[0].stats});


    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    auto const& histograms = histogram_builder_.Histograms();
    this->EvaluateSplits(histograms, m, *p_tree, {&best});
    monitor_->Stop(__func__);

    return best;
  }

  void EvaluateSplits(const common::HistCollection<GradientSumT> &hist,
                      common::GHistIndexMatrix const &gidx, const RegTree &tree,
                      std::vector<LocalExpandEntry *> entries) {
    monitor_->Start(__func__);
    const size_t grain_size = std::max<size_t>(
        1,
        column_sampler_.GetFeatureSet(tree.GetDepth(entries[0]->nid))->Size() /
        omp_get_max_threads());

    // All nodes are on the same level, so we can store the shared ptr.
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(
        entries.size());
    for (size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      auto nidx = entries[nidx_in_set]->nid;
      features[nidx_in_set] = column_sampler_.GetFeatureSet(tree.GetDepth(nidx));
    }
    common::BlockedSpace2d space(entries.size(), [&](size_t nidx_in_set) {
      return features[nidx_in_set]->Size();
    }, grain_size);

    auto num_threads = omp_get_max_threads();
    std::vector<LocalExpandEntry> tloc_candidates(omp_get_max_threads() * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      for (decltype(num_threads) j = 0; j < num_threads; ++j) {
        tloc_candidates[i * num_threads + j] = *entries[i];
      }
    }
    auto evaluator = evaluator_.GetEvaluator();

    common::ParallelFor2d(space, num_threads, [&](size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[num_threads * nidx_in_set + tidx];
      auto best = &entry->split;
      auto nidx = entry->nid;
      auto histogram = hist[nidx];
      auto features_set = features[nidx_in_set]->ConstHostSpan();
      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        if (interaction_constraints_.Query(nidx, fidx)) {
          auto grad_stats = EnumerateSplit<common::GHistRow<GradientSumT>,
                                           NodeEntry, SplitEntry, +1>(
              gidx, histogram, snode_[nidx], best, nidx, fidx, param_,
              evaluator);
          if (SplitContainsMissingValues(grad_stats, snode_[nidx])) {
            EnumerateSplit<common::GHistRow<GradientSumT>, NodeEntry,
                           SplitEntry, -1>(gidx, histogram, snode_[nidx], best,
                                           nidx, fidx, param_, evaluator);
          }
        }
      }
    });

    for (unsigned nidx_in_set = 0; nidx_in_set < entries.size();
         ++nidx_in_set) {
      for (auto tidx = 0; tidx < num_threads; ++tidx) {
        entries[nidx_in_set]->split.Update(
            tloc_candidates[num_threads * nidx_in_set + tidx].split);
      }
    }
    monitor_->Stop(__func__);
  }

  void UpdateSketch(DMatrix *m, std::vector<GradientPair> const &gpair, bst_node_t nidx) {
    auto row_set = partitioner_[nidx];
    for (auto const& page : m->GetBatches<SparsePage>()) {
      // FIXME: External memory
      for (size_t i = 0; i < row_set.Size(); ++i) {
        size_t ridx = *(row_set.begin + i);
        auto inst = page[ridx];
        sketches_.PushInstance(inst, gpair[ridx].GetHess());
      }
      common::HistogramCuts cuts;
      sketches_.MakeCuts(&cuts);
      auto row_set_span = common::Span<size_t const>{row_set.begin, row_set.Size()};
      index_.SetIndexForRowSet(page, row_set_span, cuts);
    }
  }

  void BuildNodeHistogram(const std::vector<GradientPair> &gpair,
                          common::GHistIndexMatrix const &gidx,
                          bool is_dense,
                          std::vector<bst_node_t> const &nodes_to_build) {
    histogram_builder_.BuildNodeHistogram(gpair, partitioner_.Partitions(), gidx,
                                          is_dense, nodes_to_build);
  }

  void BuildHistogram(const std::vector<GradientPair> &gpair,
                      std::vector<LocalExpandEntry> candidates,
                      common::RowSetCollection const &row_indices,
                      bool is_dense, common::GHistIndexMatrix const &gidx,
                      RegTree const* p_tree) {
    histogram_builder_.BuildHistogram(gpair, candidates, row_indices, is_dense, gidx, p_tree);
  }

 public:
  LocalApproxBuilder(TrainParam param, CPUHistMakerTrainParam hparam,
                     bst_feature_t n_features,
                     std::vector<bst_row_t> const &columns_size,
                     bool is_ranking, common::Monitor *monitor)
      : param_{param}, hist_param_{hparam}, sketches_{columns_size,
                                                      param_.max_bin,
                                                      is_ranking},
        evaluator_(param, n_features, GenericParameter::kCpuId), monitor_{
                                                                     monitor} {}

  void UpdateTree(RegTree *p_tree, DMatrix *m,
                  std::vector<GradientPair> const &gpair) {
    p_last_tree_ = p_tree;
    this->InitData(m, gpair, index_);

    auto &tree = *p_tree;

    DriverContainer<LocalExpandEntry> driver(
        static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));

    driver.Push({this->InitRoot(index_, gpair, p_tree)});
    auto num_leaves = 1;
    auto expand_set = driver.Pop();

    while (!expand_set.empty()) {
      std::vector<LocalExpandEntry> new_candidates(expand_set.size() * 2);
      // candidates that can further splited.
      std::vector<LocalExpandEntry> valid_candidates;
      // candidaates that can be applied.
      std::vector<LocalExpandEntry> applid;
      std::vector<size_t> nidx_set;
      for (size_t i = 0; i < expand_set.size(); ++i) {
        auto candidate = expand_set[i];
        if (!candidate.IsValid(param_, num_leaves)) {
          continue;
        }
        this->ApplySplit(candidate, p_tree);
        applid.push_back(candidate);
        num_leaves++;
        int left_child_nidx = tree[candidate.nid].LeftChild();
        if (LocalExpandEntry::ChildIsValid(param_, p_tree->GetDepth(left_child_nidx),
                                           num_leaves)) {
          valid_candidates.emplace_back(candidate);
          nidx_set.emplace_back(i);
        } else {
          new_candidates[i * 2] = LocalExpandEntry();
          new_candidates[i * 2 + 1] = LocalExpandEntry();
        }
      }
      this->UpdatePosition(index_, applid, p_tree);

      if (!valid_candidates.empty()) {
        this->BuildHistogram(gpair, valid_candidates, partitioner_.Partitions(),
                             m->IsDense(), index_, p_tree);
        std::vector<LocalExpandEntry*> best_splits;
        std::vector<size_t> new_candidates_pos;
        for (size_t c = 0; c < valid_candidates.size(); ++c) {
          auto i = nidx_set[c];
          auto candidate = valid_candidates[c];
          int left_child_nidx = tree[candidate.nid].LeftChild();
          int right_child_nidx = tree[candidate.nid].RightChild();
          LocalExpandEntry l_best{
              left_child_nidx, tree.GetDepth(left_child_nidx), {}};
          LocalExpandEntry r_best{
              right_child_nidx, tree.GetDepth(right_child_nidx), {}};
          new_candidates[i * 2] = l_best;
          new_candidates[i * 2 + 1] = r_best;
          best_splits.push_back(&new_candidates[i * 2]);
          best_splits.push_back(&new_candidates[i * 2 + 1]);
        }
        auto const& histograms = histogram_builder_.Histograms();
        this->EvaluateSplits(histograms, index_, tree, best_splits);
      }
      driver.Push(new_candidates.begin(), new_candidates.end());
      expand_set = driver.Pop();
    }
  }

  void UpdatePosition(common::GHistIndexMatrix const &index,
                      std::vector<LocalExpandEntry> const &candidates,
                      RegTree const *p_tree) {
    monitor_->Start(__func__);
    partitioner_.UpdatePosition(index, candidates, p_tree);
    monitor_->Stop(__func__);
  }

  void ApplySplit(LocalExpandEntry candidate, RegTree *p_tree) {
    monitor_->Start(__func__);
    ApplyTreeSplit(candidate, param_, p_tree, &snode_, &evaluator_,
                   &interaction_constraints_);
    monitor_->Stop(__func__);
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

  void InitData(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
                std::vector<GradientPair> *sampled) {
    auto const &info = m->Info();
    if (columns_size_.empty() || cached_ != m) {
      cached_ = m;
      columns_size_.resize(info.num_col_, 0);
      const auto threads = omp_get_max_threads();
      std::vector<std::vector<bst_row_t>> column_sizes(threads);
      for (auto &column : column_sizes) {
        column.resize(info.num_col_, 0);
      }
      for (auto const &page : m->GetBatches<SparsePage>()) {
        auto const &entries_per_column =
            common::HostSketchContainer::CalcColumnSize(page, info.num_col_,
                                                        threads);
        for (size_t i = 0; i < entries_per_column.size(); ++i) {
          columns_size_[i] += entries_per_column[i];
        }
      }
    }

    auto const &h_gpair = gpair->HostVector();
    sampled->resize(h_gpair.size());
    std::copy(h_gpair.cbegin(), h_gpair.cend(), sampled->begin());
    auto &rnd = common::GlobalRandom();
    if (param_.subsample != 1.0) {
      CHECK(param_.sampling_method != TrainParam::kGradientBased)
          << "Gradient based sampling is not supported for approx tree method.";
      std::bernoulli_distribution coin_flip(param_.subsample);
      std::transform(sampled->begin(), sampled->end(), sampled->begin(),
                     [&](GradientPair &g) {
                       if (coin_flip(rnd)) {
                         return g;
                       } else {
                         return GradientPair{};
                       }
                     });
    }
  }

  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
              const std::vector<RegTree *> &trees) override {
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();
    auto const &info = m->Info();

    if (hist_param_.single_precision_histogram) {
      f32_impl_ = std::make_unique<LocalApproxBuilder<float>>(
          param_, hist_param_, m->Info().num_col_, columns_size_,
          common::HostSketchContainer::UseGroup(info), &monitor_);
    } else {
      f64_impl_ = std::make_unique<LocalApproxBuilder<double>>(
          param_, hist_param_, m->Info().num_col_, columns_size_,
          common::HostSketchContainer::UseGroup(info), &monitor_);
    }

    CHECK(!param_.enable_feature_grouping)
        << "Feature grouping is not implemented for approx.";

    std::vector<GradientPair> h_gpair;
    this->InitData(gpair, m, &h_gpair);

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
