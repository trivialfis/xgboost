/**
 * Copyright 2025, XGBoost Contributors
 *
 * @brief Pure helper that turns a per-row output width (chunksize) into a prediction
 *        output shape. Kept free of any model or C-API dependency so it can be shared by
 *        the core `Learner` and the C API.
 */
#ifndef XGBOOST_COMMON_PREDICTION_SHAPE_H_
#define XGBOOST_COMMON_PREDICTION_SHAPE_H_

#include <algorithm>   // for min, max
#include <cstddef>     // for size_t
#include <cstdint>     // for uint64_t
#include <functional>  // for multiplies
#include <numeric>     // for accumulate
#include <vector>      // for vector

#include "xgboost/learner.h"  // for PredictionType
#include "xgboost/logging.h"  // for CHECK

namespace xgboost {
/**
 * @brief Determine the output shape of prediction.
 *
 * @param strict_shape Whether should we reshape the output with consideration of groups
 *                     and forest.
 * @param type         Prediction type
 * @param rows         Input samples
 * @param cols         Input features
 * @param chunksize    Total elements of output / rows
 * @param groups       Number of output groups from Learner
 * @param rounds       end_iteration - beg_iteration
 * @param out_shape    Output shape
 * @param out_dim      Output dimension
 */
inline void CalcPredictShape(bool strict_shape, PredictionType type, std::size_t rows,
                             std::size_t cols, std::size_t chunksize, std::size_t groups,
                             std::size_t rounds, std::vector<std::uint64_t> *out_shape,
                             std::uint64_t *out_dim) {
  auto &shape = *out_shape;
  if (type == PredictionType::kMargin && rows != 0) {
    // When kValue is used, softmax can change the chunksize.
    CHECK_EQ(chunksize, groups);
  }

  switch (type) {
  case PredictionType::kValue:
  case PredictionType::kMargin: {
    if (chunksize == 1 && !strict_shape) {
      *out_dim = 1;
      shape.resize(*out_dim);
      shape.front() = rows;
    } else {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      // chunksize can be 1 if it's softmax
      shape.back() = std::min(groups, chunksize);
    }
    break;
  }
  case PredictionType::kApproxContribution:
  case PredictionType::kContribution: {
    if (groups == 1 && !strict_shape) {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      shape.back() = cols + 1;
    } else {
      *out_dim = 3;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = groups;
      shape[2] = cols + 1;
    }
    break;
  }
  case PredictionType::kApproxInteraction:
  case PredictionType::kInteraction: {
    if (groups == 1 && !strict_shape) {
      *out_dim = 3;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = cols + 1;
      shape[2] = cols + 1;
    } else {
      *out_dim = 4;
      shape.resize(*out_dim);
      shape[0] = rows;
      shape[1] = groups;
      shape[2] = cols + 1;
      shape[3] = cols + 1;
    }
    break;
  }
  case PredictionType::kLeaf: {
    if (strict_shape) {
      shape.resize(4);
      shape[0] = rows;
      shape[1] = rounds;
      shape[2] = groups;
      // Guard against an empty iteration range (rounds == 0), which would otherwise
      // divide by zero. The result is multiplied by shape[1] == 0 anyway, so the value
      // is arbitrary; use 1 to keep the shape well-formed.
      auto denom = shape[1] * shape[2];
      auto forest = denom == 0 ? std::uint64_t{1}
                               : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(chunksize) / denom);
      shape[3] = forest;
      *out_dim = shape.size();
    } else if (chunksize == 1) {
      *out_dim = 1;
      shape.resize(*out_dim);
      shape.front() = rows;
    } else {
      *out_dim = 2;
      shape.resize(*out_dim);
      shape.front() = rows;
      shape.back() = chunksize;
    }
    break;
  }
  default: {
    LOG(FATAL) << "Unknown prediction type:" << static_cast<int>(type);
  }
  }
  CHECK_EQ(std::accumulate(shape.cbegin(), shape.cend(), static_cast<std::uint64_t>(1),
                           std::multiplies<>{}),
           chunksize * rows);
}
}  // namespace xgboost
#endif  // XGBOOST_COMMON_PREDICTION_SHAPE_H_
