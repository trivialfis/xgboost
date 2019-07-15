/*!
 * Copyright 2019 by Contributors
 */
#include <utility>

#include "gblinear_model.h"
#include "xgboost/json_io.h"

namespace xgboost {
namespace gbm {

class LinearReader : public JsonReader {
 public:
  explicit LinearReader(StringView str) : JsonReader(str) {}

  void ParseWeight(GBLinearModelParam const& param, std::vector<float>* p_weights) {
    size_t n_weights = (param.num_feature + 1) * param.num_output_group;
    p_weights->resize(n_weights);
    auto& weights = *p_weights;
    GetChar('[');
    for (size_t i = 0; i < n_weights; ++i) {
      char* end {nullptr};
      auto beg = raw_str_.c_str() + cursor_.Pos();

      weights[i] = std::strtof(beg, &end);

      auto len = std::distance(beg, static_cast<char const*>(end));
      for  (decltype(len) j = 0 ; j < len; ++j) { cursor_.Forward(); }

      if (i != n_weights - 1) {
        GetChar(',');
      }
    }
    GetChar(']');
  }
};

void GBLinearModel::Save(Json* p_out) const {
  auto& out = *p_out;
  out["model_param"] = toJson(param);
  std::string raw;
  raw.reserve(weight.size() * 4 /*floats*/ + weight.size() /*comma*/ );
  raw += '[';
  size_t const n_weights = weight.size();
  for (size_t i = 0; i < n_weights; ++i) {
    auto w = weight[i];
    if (i != n_weights - 1) {
      raw += std::to_string(w) + ',';
    } else {
      raw += std::to_string(w);
    }
  }
  raw += ']';
  out["linear/weights"] = JsonRaw(std::move(raw));
}

void GBLinearModel::Load(Json const& in) {
  param.InitAllowUnknown(fromJson(get<Object const>(in["model_param"])));
  auto const& raw = get<Raw>(in["linear/weights"]);
  LinearReader reader({raw.c_str(), raw.size()});
  reader.ParseWeight(param, &weight);
}

class LinearSelectRaw : public JsonReader {
  size_t* pre_pos_;

 public:
  LinearSelectRaw(StringView str, size_t* pos) :
      JsonReader{str}, pre_pos_{pos} {
    JsonReader::SetCursor(*pos);
  }

  Json ParseRaw() {
    SkipSpaces();
    auto beg = cursor_.Pos();
    while (true) {
      char ch = GetNextChar();
      while (ch != ']' && ch != -1) { ch = GetNextChar(); }
      break;
    }
    if (cursor_.Pos() == raw_str_.size()) {
      Expect(']', EOF);
    }
    auto end = cursor_.Pos();
    *pre_pos_ = cursor_.Pos();
    return Json(JsonRaw(raw_str_.substr(beg, end - beg)));
  }
};

static auto DMLC_ATTRIBUTE_UNUSED __weights_raw_parser_ = JsonReader::registry(
    "linear/weights",
    [](StringView str, size_t* pos) {
      LinearSelectRaw parser(str, pos);
      auto ret = parser.ParseRaw();
      return ret;
    });

}  // namespace gbm
}  // namespace xgboost
