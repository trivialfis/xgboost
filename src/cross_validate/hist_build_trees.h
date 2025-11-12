/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once
#include <memory>
#include <vector>

#include "xgboost/context.h"
#include "xgboost/data.h"
#include "xgboost/gradient.h"
#include "xgboost/tree_model.h"

namespace xgboost::cv {
void BuildTrees(Context const* ctx, DMatrix* p_fmat,
                std::vector<std::vector<std::unique_ptr<GradientContainer>>> const& gpairs,
                std::vector<std::vector<std::vector<bst_idx_t>>> const& tr_idx,
                std::vector<RegTree*> trees);
}
