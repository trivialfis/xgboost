#include <gtest/gtest.h>

#include "../../../src/common/numa_utils.h"

namespace xgboost::common {
TEST(Numa, GetCpus) {
  std::vector<std::int32_t> cpus;
  GetNumaNodeCpus(&cpus);
}
}  // namespace xgboost::common
