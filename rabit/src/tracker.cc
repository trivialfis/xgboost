#include <thread>
#include <map>
#include "rabit/internal/socket.h"

namespace rabit {
class Worker {
 public:
  enum Action {
    kPrint,
    kStart,
    kShutdown
  };

 private:
  Action act_;
  int32_t rank_;
  std::string job_id_;
  utils::TCPSocket socket_;

 public:
  Action CurAction() const { return act_; }
  int32_t GetRank() const { return rank_;}

  void AssignRank(int32_t rank) {
    // see all reduce base ReConnectLinks
    socket_.Send(&rank, sizeof(rank));
    int32_t parent_rank = 0;
    socket_.Send(&parent_rank, sizeof(parent_rank));
    int32_t world_size = 0;
    socket_.Send(&world_size, sizeof(world_size));
  }
};

class Tracker {
  utils::TCPSocket socket_;
  int32_t n_workers_;
  std::string host_;
  int32_t port_;
  std::thread t_;

 public:
  Tracker(std::string host, int32_t n_workers, int32_t port)
      : n_workers_{n_workers}, port_{port} {
    socket_.Listen(256);
  }

  std::vector<int32_t> GetNeighbors() const {
    std::vector<int32_t> neighbors;
    return neighbors;
  }

  void GetTree();

  void FindSharedRing();

  void GetRing();

  void GetLinkMap();

  void AcceptWorkers(int32_t n_workers) {
    std::vector<int32_t> shutdown;
    std::vector<Worker> pending;

    std::map<int32_t, int32_t> tree_map;
    while (shutdown.size() != n_workers) {
      this->socket_.Accept();
      auto w = Worker{};
      if (w.CurAction() == Worker::kPrint) {
        std::string msg;
        socket_.RecvStr(&msg);
        continue;
      } else if (w.CurAction() == Worker::kShutdown) {
        shutdown.push_back(w.GetRank());
        continue;
      } else {
        GetLinkMap();
      }

      int32_t rank = 0;  // fixme

      if (rank == -1) {
        pending.push_back(w);
      }
    }
  }

  void Start(int32_t n_workers) {
    t_ = std::thread{[=]() { this->AcceptWorkers(n_workers); }};
  }

  void Stop() {
    if (t_.joinable()) {
      t_.join();  // fixme: timeout
    }
  }
};
}  // namespace rabit
