#pragma once

#include <cstdint>  // for int32_t

#if defined(_WIN32)
#define AF_INET 2    // internetwork: UDP, TCP, etc.
#define AF_INET6 23  // Internetwork Version 6
#else
#include <sys/socket.h>  // for AF_INET, AF_INET6
#endif

namespace xgboost::collective {
enum class SockDomain : std::int32_t { kV4 = AF_INET, kV6 = AF_INET6 };
}  // namespace namespace xgboost::collective
