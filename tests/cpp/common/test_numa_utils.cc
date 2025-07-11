/**
 * Copyright 2025, by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <filesystem>  // for path
#include <fstream>     // for ofstream
#include <vector>      // for vector

#include "../../../src/common/numa_utils.h"
#include "../filesystem.h"

namespace xgboost::common {
namespace {
namespace fs = std::filesystem;
}

TEST(Numa, CpuListParser) {
  dmlc::TemporaryDirectory tmpdir;
  auto path = fs::path{tmpdir.path} / "cpulist";
  std::vector<std::int32_t> cpus;

  auto write = [&](auto const& cpulist) {
    std::ofstream fout{path};
    fout << cpulist;
  };

  {
    std::string cpulist = R"(1
)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    ASSERT_EQ(cpus[0], 1);
    ASSERT_EQ(cpus.size(), 1);
  }
  {
    std::string cpulist = R"(2)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    ASSERT_EQ(cpus.size(), 1);
    ASSERT_EQ(cpus[0], 2);
  }
  {
    std::string cpulist = R"(2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    ASSERT_EQ(cpus.size(), 2);
    ASSERT_EQ(cpus[0], 2);
    ASSERT_EQ(cpus[1], 3);
  }

  auto check_4cpu_case = [&] {
    ASSERT_EQ(cpus.size(), 4);
    for (std::size_t i = 0; i < cpus.size(); ++i) {
      ASSERT_EQ(cpus[i], static_cast<std::int32_t>(i));
    }
  };
  {
    std::string cpulist = R"(0-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0-2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1-2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1,2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0,1,2-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0-1,2,3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
  {
    std::string cpulist = R"(0-1,2-3)";
    write(cpulist);
    ReadCpuList(path, &cpus);
    check_4cpu_case();
  }
}

TEST(Numa, GetCpus) {
  std::vector<std::int32_t> cpus;
  GetNumaNodeCpus(&cpus);
  ASSERT_FALSE(cpus.empty());
}

TEST(Numa, GetMaxNodes) {
  auto n_nodes = GetMaxNumNodes();
  ASSERT_GE(n_nodes, sizeof(std::uint64_t) * 8);
}

TEST(Numa, GetMemBind) { [[maybe_unused]] auto bind = GetNumaMemBind(); }
}  // namespace xgboost::common
