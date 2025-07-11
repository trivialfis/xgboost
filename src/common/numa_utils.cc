/**
 * Copyright 2025, by XGBoost Contributors
 */
#include <linux/mempolicy.h>  // for MPOL_BIND
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>     // for int32_t
#include <filesystem>  // for path
#include <fstream>     // for ifstream
#include <string>      // for string
#include <vector>      // for vector

#include "common.h"  // for TrimLast
#include "cuda_rt_utils.h"
#include "error_msg.h"  // for SystemError
#include "xgboost/logging.h"

namespace xgboost::common {
namespace {
namespace fs = std::filesystem;

// Wrapper for the system call
auto GetMemPolicy(int *mode, unsigned long *nodemask, unsigned long maxnode, void *addr,
                  unsigned long flags) {
  return syscall(SYS_get_mempolicy, mode, nodemask, maxnode, addr, flags);
}
}  // namespace

void ReadCpuList(fs::path const &path, std::vector<std::int32_t> *p_cpus) {
  auto &cpus = *p_cpus;
  cpus.clear();

  std::ifstream fin{path};
  std::string buff;
  fin >> buff;
  CHECK(!buff.empty());
  buff = common::TrimFirst(common::TrimLast(buff));

  std::int32_t k = 0;
  CHECK(std::isalnum(buff[k]));
  while (static_cast<std::size_t>(k) < buff.size()) {
    std::int32_t val0 = -1, val1 = -1;
    std::size_t idx = 0;
    CHECK(std::isalnum(buff[k])) << k << " " << buff;
    val0 = std::stoi(buff.data() + k, &idx);
    auto last = k + idx;
    CHECK_LE(last, buff.size());
    k = last + 1;  // new begin
    if (last == buff.size() || buff[last] != '-') {
      cpus.push_back(val0);
      continue;
    }
    CHECK_EQ(buff[last], '-') << last;

    idx = -1;
    CHECK_LT(k, buff.size());
    val1 = std::stoi(buff.data() + k, &idx);
    CHECK_GE(idx, 1);
    // Parse range
    for (auto i = val0; i <= val1; ++i) {
      cpus.push_back(i);
    }
    k += (idx + 1);
  }
}

void GetNumaNodeCpus(std::vector<std::int32_t> *p_cpus) {
  std::int32_t nodeid = curt::GetNumaId();
  std::string nodename = "node" + std::to_string(nodeid);
  auto p_cpulist = fs::path{"/sys/devices/system/node"} / fs::path{nodename} / fs::path{"cpulist"};
  p_cpus->clear();
  if (!fs::exists(p_cpulist)) {
    return;
  }
  ReadCpuList(p_cpulist, p_cpus);
}

auto GetMemPolicy(int *policy, unsigned long *nmask, unsigned long maxnode) {
  return GetMemPolicy(policy, nmask, maxnode, nullptr, 0);
}

[[nodiscard]] std::size_t GetMaxNumNodes() {
  auto p_possible = fs::path{"/sys/devices/system/node/possible"};
  std::int32_t max_n_nodes = -1;
  if (fs::exists(p_possible)) {
    std::vector<std::int32_t> cpus;
    ReadCpuList(p_possible, &cpus);
    auto it = std::max_element(cpus.cbegin(), cpus.cend());
    if (it != cpus.cend()) {
      max_n_nodes = *it;
    }
  }

  if (max_n_nodes <= 0) {
    max_n_nodes = sizeof(uint64_t) * 8;
  }
  while (true) {
    std::vector<std::uint64_t> mask(max_n_nodes, 0);

    std::int32_t mode = -1;
    auto err = GetMemPolicy(&mode, mask.data(), max_n_nodes);
    if (!err || errno != EINVAL) {
      return max_n_nodes;
    }
    max_n_nodes *= 2;
  }
}

[[nodiscard]] bool GetNumaMemBind() {
  std::int32_t mode = -1;
  auto max_n_nodes = GetMaxNumNodes();
  std::vector<std::uint64_t> mask(max_n_nodes / 8);
  CHECK_GE(GetMemPolicy(&mode, mask.data(), max_n_nodes), 0) << error::SystemError().message();
  return mode == MPOL_BIND;
}
}  // namespace xgboost::common
