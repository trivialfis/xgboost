#include <linux/mempolicy.h>  // for MPOL_BIND
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>   // for find
#include <cstdint>     // for int32_t
#include <filesystem>  // for path
#include <fstream>     // for ifstream
#include <string>      // for string
#include <vector>      // for vector

#include "cuda_rt_utils.h"
#include "error_msg.h"  // for SystemError
#include "xgboost/logging.h"

namespace xgboost::common {
namespace {
namespace fs = std::filesystem;

// Read a file with the `cpulist` format.
void ReadCpuList(fs::path const &path, std::vector<std::int32_t> *p_cpus) {
  auto n_bytes = fs::file_size(path);
  std::ifstream fin{path};
  std::string buffer(n_bytes, 0);
  fin.read(buffer.data(), buffer.size());
  auto pos = std::find(buffer.cbegin(), buffer.cend(), '-');
  if (pos == buffer.cend()) {
    return;
  }
  auto start = std::stoi(buffer.substr(0, std::distance(buffer.cbegin(), pos)));
  auto end = std::stoi(buffer.substr(std::distance(buffer.cbegin(), pos) + 1));

  auto &cpus = *p_cpus;
  for (auto i = start; i <= end; ++i) {
    cpus.push_back(i);
  }
}

// Wrapper for the system call
auto GetMemPolicy(int *mode, unsigned long *nodemask, unsigned long maxnode, void *addr,
                  unsigned long flags) {
  return syscall(SYS_get_mempolicy, mode, nodemask, maxnode, addr, flags);
}
}  // namespace

void GetNumaNodeCpus(std::vector<std::int32_t> *p_cpus) {
  std::int32_t nodeid = curt::GetNumaId();
  std::string nodename = "node" + std::to_string(nodeid);
  auto cpulist_p = fs::path{"/sys/devices/system/node"} / fs::path{nodename} / fs::path{"cpulist"};
  if (!fs::exists(cpulist_p)) {
    return;
  }
  ReadCpuList(cpulist_p, p_cpus);
}

auto GetMemPolicy(int *policy, unsigned long *nmask, unsigned long maxnode) {
  return GetMemPolicy(policy, nmask, maxnode, nullptr, 0);
}

[[nodiscard]] std::size_t GetMaxNumNodes() {
  auto p_possible = fs::path{"/sys/devices/system/node/possible"};
  if (!fs::exists(p_possible)) {
    return 0;
  }

  std::int32_t max_num_nodes = sizeof(uint64_t) * 8;
  while (true) {
    std::vector<std::uint64_t> mask(max_num_nodes);

    std::int32_t mode = -1;
    auto err = GetMemPolicy(&mode, mask.data(), max_num_nodes);
    if (!err || errno != EINVAL) {
      return max_num_nodes;
    }
    max_num_nodes *= 2;
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
