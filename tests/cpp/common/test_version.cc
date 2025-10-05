/**
 * Copyright 2019-2025, XGBoost contributors
 */
#include <dmlc/io.h>
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/json.h>
#include <xgboost/version_config.h>

#include <string>

#include "../../../src/common/version.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory

namespace xgboost {
TEST(Version, Basic) {
  Json j_ver { Object() };
  Version::Save(&j_ver);
  auto triplet { Version::Load(j_ver) };
  ASSERT_TRUE(Version::Same(triplet));

  common::TemporaryDirectory tempdir;
  const std::string fname = tempdir.Str() + "/version";

  {
    auto fo = std::make_unique<common::AlignedFileWriteStream>(fname, "w");
    Version::Save(fo.get());
  }

  {
    auto fi = std::make_unique<common::AlignedFileReadStream>(fname);
    auto triplet{Version::Load(fi.get())};
    ASSERT_TRUE(Version::Same(triplet));
  }

  std::string str { Version::String(triplet) };

  size_t ptr {0};
  XGBoostVersionT v {0};
  v = std::stoi(str, &ptr);
  ASSERT_EQ(str.at(ptr), '.');
  ASSERT_EQ(v, XGBOOST_VER_MAJOR) << "major: " << v;

  str = str.substr(ptr+1);

  ptr = 0;
  v = std::stoi(str, &ptr);
  ASSERT_EQ(str.at(ptr), '.');
  ASSERT_EQ(v, XGBOOST_VER_MINOR) << "minor: " << v;;

  str = str.substr(ptr+1);

  ptr = 0;
  v = std::stoi(str, &ptr);
  ASSERT_EQ(v, XGBOOST_VER_PATCH) << "patch: " << v;;

  str = str.substr(ptr);
  ASSERT_EQ(str.size(), 0);
}
}  // namespace xgboost
