#pragma once
#include <cstdint>  // for int32_t
#include <vector>   // for vector

namespace xgboost::common {
void GetNumaNodeCpus(std::vector<std::int32_t>* p_cpus);

[[nodiscard]] std::size_t GetMaxNumNodes();

[[nodiscard]] bool GetNumaMemBind();
}  // namespace xgboost::common
