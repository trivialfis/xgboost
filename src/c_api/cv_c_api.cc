/**
 * Copyright 2025, XGBoost contributors
 */
#include "../cross_validate/hist_build_trees.h"
#include "../data/array_interface.h"
#include "../gbm/gbtree_model.h"
#include "c_api_error.h"
#include "xgboost/json.h"
#include "xgboost/predictor.h"

using namespace xgboost;  // NOLINT

namespace xgboost::gbm {
struct GBTreeCvFolds {
  std::vector<std::shared_ptr<GBTreeModel>> folds;
};
}  // namespace xgboost::gbm

typedef void* GBTreeCvFoldsHandle;   // NOLINT
typedef void* GBTreeModelHandle;     // NOLINT
typedef void* ArrayContainerHandle;  // NOLINT

XGB_DLL int XGCvUpdateOneIter(GBTreeCvFoldsHandle handle, DMatrixHandle fmat,
                              char const* tr_indices, char const* grad, char const* hess) {
  using BatchTrIdx = std::vector<std::vector<bst_idx_t>>;

  API_BEGIN();
  CHECK_HANDLE();

  auto p_fmat = CastDMatrixHandle(fmat);

  auto jindices = Json::Load(StringView{tr_indices});
  auto const& jindices_array = get<Array const>(jindices);
  std::size_t n_batches = jindices_array.size();

  std::int32_t n_folds = 0;
  std::vector<BatchTrIdx> tr_idx;
  for (std::size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    auto const& batch = get<Array const>(jindices[batch_idx]);
    if (n_folds == 0) {
      n_folds = batch.size();
    }
    CHECK_EQ(n_folds, batch.size());
    BatchTrIdx batch_tr_idx;
    for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto const& jfold = get<Object const>(batch[fold_idx]);
      auto fold = ArrayInterface<1>{jfold};
      batch_tr_idx.emplace_back();
      auto& fold_tr_idx = batch_tr_idx.back();
      DispatchDType(fold, DeviceOrd::CPU(), [&](auto&& in) {
        for (std::size_t i = 0; i < in.template Shape<0>(); ++i) {
          fold_tr_idx.push_back(in(i));
        }
      });
    }
    CHECK_EQ(batch_tr_idx.size(), n_folds);
    tr_idx.emplace_back(std::move(batch_tr_idx));
  }
  CHECK_EQ(tr_idx.size(), n_batches);
  std::cout << "n_batches:" << n_batches << " n_folds:" << n_folds << std::endl;

  // Load gradient
  auto jgrad = Json::Load(grad);
  auto jhess = Json::Load(hess);

  auto const& jgrad_array = get<Array const>(jgrad);
  CHECK_EQ(jgrad_array.size(), n_batches);
  auto const& jhess_array = get<Array const>(jhess);
  CHECK_EQ(jhess_array.size(), n_batches);

  bst_target_t n_targets = 0;  // fixme

  std::vector<std::vector<std::unique_ptr<GradientContainer>>> gpairs;
  for (std::size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    auto const& batch_grad = get<Array const>(jgrad_array[batch_idx]);
    CHECK_EQ(batch_grad.size(), n_folds);
    auto const& batch_hess = get<Array const>(jhess_array[batch_idx]);
    CHECK_EQ(batch_hess.size(), n_folds);
    std::vector<std::unique_ptr<GradientContainer>> batch_gpairs;
    for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
      auto fold_grad = ArrayInterface<2>{get<Object const>(batch_grad[fold_idx])};
      auto fold_hess = ArrayInterface<2>{get<Object const>(batch_hess[fold_idx])};

      auto fold_gpair = std::make_unique<GradientContainer>();
      fold_gpair->gpair.Reshape(fold_grad.Shape<0>(), fold_grad.Shape<1>());
      if (n_targets == 0) {
        n_targets = fold_gpair->NumTargets();
      }
      CHECK_EQ(n_targets, fold_gpair->NumTargets());

      auto& h_gpair = fold_gpair->gpair.Data()->HostVector();
      CHECK_EQ(h_gpair.size(), fold_grad.n);
      for (std::size_t i = 0; i < h_gpair.size(); ++i) {
        h_gpair[i] = GradientPair{fold_grad(i), fold_hess(i)};
      }
      batch_gpairs.emplace_back(std::move(fold_gpair));
    }
    CHECK_EQ(batch_gpairs.size(), n_folds);
    gpairs.emplace_back(std::move(batch_gpairs));
  }

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "cuda"}});

  std::vector<std::unique_ptr<RegTree>> trees;
  for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    trees.emplace_back(std::make_unique<RegTree>(1, p_fmat->Info().num_col_));
  }
  std::vector<RegTree*> p_trees;
  std::transform(trees.begin(), trees.end(), std::back_inserter(p_trees),
                 [](auto& t) { return t.get(); });

  cv::BuildTrees(&ctx, p_fmat.get(), gpairs, tr_idx, p_trees);

  // fixme
  LearnerModelParam lparam;
  lparam.num_feature = p_fmat->Info().num_col_;
  CHECK_GE(n_targets, 1);
  lparam.num_output_group = n_targets;

  auto p_folds = static_cast<gbm::GBTreeCvFolds*>(handle);
  for (decltype(n_folds) fold_idx = 0; fold_idx < n_folds; ++fold_idx) {
    auto model = std::make_shared<gbm::GBTreeModel>(&lparam, &ctx);
    // fixme
    auto& tree = trees[fold_idx];
    gbm::TreesOneGroup group_trees;
    group_trees.emplace_back(std::move(tree));
    gbm::TreesOneIter fold_trees;
    fold_trees.emplace_back(std::move(group_trees));
    model->CommitModel(std::move(fold_trees));
    p_folds->folds.at(fold_idx) = model;
  }

  API_END();
}

XGB_DLL int XGCvFoldsCreate(char const* config, GBTreeCvFoldsHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(config);
  auto n_folds = get<Integer const>(Json::Load(config)["n_folds"]);
  auto p_folds = new gbm::GBTreeCvFolds{};
  p_folds->folds.resize(n_folds);
  *out = p_folds;
  API_END();
}

XGB_DLL int XGCvFoldsFree(GBTreeCvFoldsHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  auto p_folds = static_cast<gbm::GBTreeCvFolds*>(handle);
  delete p_folds;
  API_END();
}

XGB_DLL int XGCvGetFold(GBTreeCvFoldsHandle handle, int fold_idx, GBTreeModelHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(handle);
  auto* folds = static_cast<gbm::GBTreeCvFolds*>(handle);
  auto n_folds = folds->folds.size();
  CHECK_LE(fold_idx, n_folds);
  auto fold = folds->folds[fold_idx];
  xgboost_CHECK_C_ARG_PTR(out);
  *out = new std::shared_ptr<gbm::GBTreeModel>{fold};
  API_END();
}

XGB_DLL int XGGBTreeModelFree(GBTreeModelHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  auto fold = static_cast<std::shared_ptr<gbm::GBTreeModel>*>(handle);
  delete fold;
  API_END();
}

struct ArrayContainer {
  linalg::Tensor<float> data;
  std::int32_t kdims{0};
};

XGB_DLL int XGArrayContainerCreate(char const* config, ArrayContainerHandle* out) {
  API_BEGIN();
  auto jconf = Json::Load(config);
  auto jshape = get<Array const>(jconf["shape"]);
  auto n_dims = jshape.size();
  constexpr auto kFullDim = decltype(std::declval<ArrayContainer>().data)::kDimension;
  std::vector<std::size_t> shape(kFullDim, 1);
  for (std::size_t k = 0; k < n_dims; ++k) {
    shape[k] = get<Integer const>(jshape[k]);
  }
  auto array = new ArrayContainer;
  array->data.Reshape(common::Span<std::size_t const, kFullDim>{shape});
  array->kdims = n_dims;
  *out = array;
  API_END();
}

XGB_DLL int XGArrayContainerFree(ArrayContainerHandle handle) {
  API_BEGIN();
  CHECK_HANDLE();
  auto array = static_cast<ArrayContainer*>(handle);
  delete array;
  API_END();
}

XGB_DLL int XGGBTreeModelPredict(GBTreeModelHandle handle, DMatrixHandle fmat,
                                 ArrayContainerHandle c_array) {
  API_BEGIN();
  CHECK_HANDLE();
  auto fold = static_cast<std::shared_ptr<gbm::GBTreeModel>*>(handle);
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "cuda"}});
  auto predictor = std::unique_ptr<Predictor>{Predictor::Create("gpu_predictor", &ctx)};

  auto p_fmat = CastDMatrixHandle(fmat);
  PredictionCacheEntry out_prediction;
  predictor->InitOutPredictions(p_fmat->Info(), &out_prediction.predictions, **fold);

  predictor->PredictBatch(p_fmat.get(), &out_prediction, **fold, 0);

  // fixme: less copies
  auto array = static_cast<ArrayContainer*>(c_array);
  CHECK_EQ(array->data.Size(), out_prediction.predictions.Size());
  array->data.SetDevice(out_prediction.predictions.Device());
  // fixme: set shape here
  array->data.ModifyInplace(
      [&](HostDeviceVector<float>* data, auto) { data->Copy(out_prediction.predictions); });
  API_END();
}
