/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t
#include <string>   // for string
#include <utility>  // for move

#include "xgboost/collective/result.h"  // for Result
#include "xgboost/json.h"               // for Json

namespace xgboost::collective {
class TCPSocket;
}

namespace xgboost::collective::proto {
struct PeerInfo {
  std::string host;
  std::int32_t port{-1};
  std::int32_t rank{-1};

  PeerInfo() = default;
  PeerInfo(std::string host, std::int32_t port, std::int32_t rank)
      : host{std::move(host)}, port{port}, rank{rank} {}

  explicit PeerInfo(Json const& peer)
      : host{get<String>(peer["host"])},
        port{static_cast<std::int32_t>(get<Integer const>(peer["port"]))},
        rank{static_cast<std::int32_t>(get<Integer const>(peer["rank"]))} {}

  [[nodiscard]] Json ToJson() const {
    Json info{Object{}};
    info["rank"] = rank;
    info["host"] = String{host};
    info["port"] = Integer{port};
    return info;
  }

  [[nodiscard]] auto HostPort() const { return host + ":" + std::to_string(this->port); }
};

struct Magic {
  static constexpr std::int32_t kMagic = 0xff99;

  [[nodiscard]] Result Verify(xgboost::collective::TCPSocket* p_sock);
};

// Basic commands for communication between workers and the tracker.
enum class CMD : std::int32_t {
  kInvalid = 0,
  kStart = 1,
  kShutdown = 2,
  kError = 3,
  kPrint = 4,
};

struct Connect {
  [[nodiscard]] Result WorkerSend(TCPSocket* tracker, std::int32_t world, std::int32_t rank,
                                  std::string task_id) const;
  [[nodiscard]] Result TrackerRecv(TCPSocket* sock, std::int32_t* world, std::int32_t* rank,
                                   std::string* task_id) const;
};

class Start {
 private:
  [[nodiscard]] Result TrackerSend(std::int32_t world, TCPSocket* worker) const;

 public:
  [[nodiscard]] Result WorkerSend(std::int32_t lport, TCPSocket* tracker, std::int32_t eport) const;
  [[nodiscard]] Result WorkerRecv(TCPSocket* tracker, std::int32_t* p_world) const;
  [[nodiscard]] Result TrackerHandle(Json jcmd, std::int32_t* recv_world, std::int32_t world,
                                     std::int32_t* p_port, TCPSocket* p_sock,
                                     std::int32_t* eport) const;
};

// Protocol for communicating with the tracker for printing message.
struct Print {
  [[nodiscard]] Result WorkerSend(TCPSocket* tracker, std::string msg) const;
  [[nodiscard]] Result TrackerHandle(Json jcmd, std::string* p_msg) const;
};

// Protocol for communicating with the tracker during error.
struct ErrorCMD {
  [[nodiscard]] Result WorkerSend(TCPSocket* tracker, Result const& res) const;
  [[nodiscard]] Result TrackerHandle(Json jcmd, std::string* p_msg, int* p_code) const;
};

// Protocol for communicating with the tracker during shutdown.
struct ShutdownCMD {
  [[nodiscard]] Result Send(TCPSocket* peer) const;
};

// Protocol for communicating with the local error handler during error or shutdown. Only
// one protocol that doesn't have the tracker involved.
struct Error {
  constexpr static std::int32_t ShutdownSignal() { return 0; }
  constexpr static std::int32_t ErrorSignal() { return -1; }

  [[nodiscard]] Result SignalError(TCPSocket* worker) const;
  // self is localhost, we are sending the signal to the error handling thread for it to
  // close.
  [[nodiscard]] Result SignalShutdown(TCPSocket* self) const;
  // get signal, either for error or for shutdown.
  [[nodiscard]] Result RecvSignal(TCPSocket* peer, bool* p_is_error) const;
};
}  // namespace xgboost::collective::proto
