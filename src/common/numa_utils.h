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

[[nodiscard]] std::size_t GetMaxNumNodes();

[[nodiscard]] bool GetNumaMemBind();
}  // namespace xgboost::common
