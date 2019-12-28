#include <queue>

#include "xgboost/tree_updater.h"
#include "xgboost/tree_model.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"

#include "param.h"

namespace xgboost {

using ShapeType = std::array<size_t, 4>;

class MatrixView {
 protected:
  common::Span<float> data_;
  ShapeType shape_;
  ShapeType stride_;

 public:
  MatrixView(common::Span<float> data, ShapeType shape, ShapeType stride) :
      data_{data}, shape_{shape}, stride_{stride} {}

  ShapeType Shape() const {
    return shape_;
  }
  ShapeType Strides() const {
    return stride_;
  }
};

class VectorView {
  common::Span<float> data_;
  size_t stride_;

 public:
  VectorView(common::Span<float> d, size_t stride) : data_{d}, stride_{stride} {}
  size_t Size() const { return data_.size(); }
  size_t Stride() const { return stride_; }

  VectorView& operator+=(VectorView const& that) {
    CHECK_EQ(data_.size(), that.Size());
    for (size_t i = 0; i < data_.size(); i += stride_) {
      data_[i] += that.data_[i];
    }
  }

  float const& operator[](size_t i) const {
    return data_[i * stride_];
  }
  float& operator[](size_t i) {
    return data_[i * stride_];
  }
};

class Matrix : public MatrixView {
  HostDeviceVector<float> storage_;

 public:
  static constexpr size_t kAll = std::numeric_limits<size_t>::max();

  VectorView Row(size_t i) {
    auto& h_storage = storage_.HostVector();
    return {
      common::Span<float>{h_storage.data() + i * shape_[1], shape_[1]},
      1
    };
  }
};


namespace tree {

class MutliQuantileHist : public TreeUpdater {
 public:
  void Configure(const Args& args) override {}
  void LoadConfig(Json const& in) override {}
  void SaveConfig(Json* p_out) const override {}
  char const* Name() const override {
    return "multi_histogram";
  }

  void ApplySplit();
  void EvaluateSplitsExact(bst_node_t node_id, DMatrix* fmat,
                           HostDeviceVector<GradientPair>* gradients) {
    SplitEntry split;
    for (auto const& page : fmat->GetBatches<SortedCSCPage>()) {
      for (size_t i = 0; i < page.Size(); ++i) {
        common::Span<Entry const> column = page[i];
        std::vector<float> sum_gradient (n_targets_);
        std::vector<float> sum_hessian (n_targets_);
        auto const& h_gradient = gradients->HostVector();
        for (size_t j = 0; j < column.size(); ++j) {
          for (size_t k = 0; k < n_targets_; ++k) {
            sum_gradient[k] += h_gradient[j * n_targets_ + k].GetGrad();
            sum_hessian[k] += h_gradient[j * n_targets_ + k].GetHess();
          }
        }
      }
    }
  }

  void InitRoot(DMatrix* fmat, HostDeviceVector<GradientPair>* gpair, RegTree* p_tree) {
    auto const& h_gpair = gpair->ConstHostVector();
    sum_gradients_.resize(1 * n_targets_);
    sum_hessians_.resize(1 * n_targets_);
    for (size_t i = 0; i < gpair->Size(); i += 1) {
      for (size_t j = 0; j < n_targets_; ++j) {
        sum_gradients_[j] += h_gpair[i + j].GetGrad();
      }
      for (size_t j = 0; j < n_targets_; ++j) {
        sum_hessians_[j] += h_gpair[i + j].GetHess();
      }
    }

    row_index_.resize(fmat->Info().num_row_);
    positions_.emplace_back(Segment{0, fmat->Info().num_row_});
  }

  void UpdateTree(HostDeviceVector<GradientPair>* gpair,
                  DMatrix* data,
                  RegTree* p_tree) {
    this->InitRoot(data, gpair, p_tree);
    while(!expand_queue_.empty()) {
      bst_node_t node = expand_queue_.front();
      expand_queue_.pop();
    }
  }

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* data,
              const std::vector<RegTree*>& trees) override {
    for (auto p_tree : trees) {
      this->UpdateTree(gpair, data, p_tree);
    }
  }

 private:
  std::vector<float> sum_gradients_;
  std::vector<float> sum_hessians_;

  struct Segment {
    size_t beg;
    size_t size;

    Segment(size_t beg, size_t size): beg {beg}, size{size} {}
  };

  std::vector<Segment> positions_;
  std::vector<bst_row_t> row_index_;

  TrainParam param_;
  std::queue<bst_node_t> expand_queue_;
  bst_feature_t n_targets_ {1};
};

XGBOOST_REGISTER_TREE_UPDATER(MutliQuantileHist, "multi_histogram")
    .describe("Grow tree with multi-output leaf.")
    .set_body([]() { return new MutliQuantileHist(); });

}  // namespace tree
}  // namespace xgboost
