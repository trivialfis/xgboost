#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>   // for find
#include <cstdint>     // for int32_t
#include <filesystem>  // for path
#include <fstream>     // for ifstream
#include <string>      // for string
#include <vector>      // for vector

#include "cuda_rt_utils.h"

namespace xgboost::common {
void GetNumaNodeCpus(std::vector<std::int32_t>* p_cpus) {
  namespace fs = std::filesystem;
  std::int32_t nodeid = curt::GetNumaId();
  std::string nodename = "node" + std::to_string(nodeid);
  auto cpulist_p = fs::path{"/sys/devices/system/node"} / fs::path{nodename} / fs::path{"cpulist"};
  if (!fs::exists(cpulist_p)) {
    return;
  }

  auto n_bytes = fs::file_size(cpulist_p);
  std::ifstream fin{cpulist_p};
  std::string buffer(n_bytes, 0);
  fin.read(buffer.data(), buffer.size());
  auto pos = std::find(buffer.cbegin(), buffer.cend(), '-');
  if (pos == buffer.cend()) {
    return;
  }
  auto start = std::stoi(buffer.substr(0, std::distance(buffer.cbegin(), pos)));
  auto end = std::stoi(buffer.substr(std::distance(buffer.cbegin(), pos) + 1));

  auto& cpus = *p_cpus;
  for (auto i = start; i <= end; ++i) {
    cpus.push_back(i);
  }
}

void SetNumaCpuAffinity(std::vector<std::int32_t> const& cpus) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (auto cpu_id : cpus) {
    CPU_SET(cpu_id, &cpuset);
  }
  // fixme: is there a way to set for the entire process?
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == -1) {
    std::perror("sched_setaffinity");
    // fixme, error handling.
  }
}

void SetNumaMemoryAffinity(std::int32_t node_id) {
  std::uintmax_t nodemask = 1UL << node_id;
  int mode = MPOL_BIND;
  auto maxnode = sizeof(nodemask) * 8;  // fixme: not true

  auto ret = syscall(SYS_set_mempolicy, mode, &nodemask, maxnode);
  if (ret != 0) {
    std::perror("set_mempolicy");
    // fixme: error handling.
  }
}
}  // namespace xgboost::common
