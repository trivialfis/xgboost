#include <gtest/gtest.h>
#include <algorithm>
#include "xgboost/span.h"
#include "../../../src/common/selection.h"


namespace xgboost {

template <typename T>
void TestSelectionSinlge(std::vector<T> inputs, size_t order) {
  ASSERT_LE(order, inputs.size());
  std::vector<T> copy { inputs };
  std::vector<T> copy_1 { inputs };

  Select(0, inputs.begin(), inputs.begin() + order, inputs.end());

  std::stable_sort(copy.begin(), copy.end());
  EXPECT_EQ(inputs[order], copy[order]);
}

TEST(Selection, Single) {
  std::vector<float> inputs {2, 1, 2, 2, 0, 9, 3.3, 7, 19, 19, 0, 0, 0, 1, 1};
  for (size_t i = 0; i < inputs.size(); ++i) {
    TestSelectionSinlge(inputs, i);
  }
}

}  // namespace xgboost
