/**
 * Copyright 2025, by XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t
#include <filesystem>
#include <vector>  // for vector

namespace xgboost::common {
/** @brief Read a file with the `cpulist` format. */
void ReadCpuList(std::filesystem::path const &path, std::vector<std::int32_t> *p_cpus);

void GetNumaNodeCpus(std::vector<std::int32_t> *p_cpus);

/**
 * @brief Find the maximum number of NUMA nodes.
 */
[[nodiscard]] std::int32_t GetMaxNumNodes();

/**
 * @brief Check whether the memory policy is set to bind.
 */
[[nodiscard]] bool GetNumaMemBind();
}  // namespace xgboost::common
