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
}  // namespace xgboost::common
