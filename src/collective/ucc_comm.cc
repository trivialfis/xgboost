#include <ucc/api/ucc.h>

#include <cstdint>

namespace xgboost::collective {
void InitUcc() {
  ucc_lib_h lib;
  ucc_lib_config_h lib_config;
  ucc_lib_params_t lib_params;

  std::uint32_t major = 0, minor = 0, release = 0;
  ucc_get_version(&major, &minor, &release);
}
}  // namespace xgboost::collective
