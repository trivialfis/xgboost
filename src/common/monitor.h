/*!
 * Copyright by Contributors 2017
 */
#pragma once
#include <xgboost/logging.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>

#if defined(XGBOOST_USE_NVTX) && defined(__CUDACC__)
#include <nvToolsExt.h>
#endif  // defined(XGBOOST_USE_NVTX) && defined(__CUDACC__)

namespace xgboost {
namespace common {
struct Timer {
  using ClockT = std::chrono::high_resolution_clock;
  using TimePointT = std::chrono::high_resolution_clock::time_point;
  using DurationT = std::chrono::high_resolution_clock::duration;
  using SecondsT = std::chrono::duration<double>;

  TimePointT start;
  DurationT elapsed;
  Timer() { Reset(); }
  void Reset() {
    elapsed = DurationT::zero();
    Start();
  }
  void Start() { start = ClockT::now(); }
  void Stop() { elapsed += ClockT::now() - start; }
  double ElapsedSeconds() const { return SecondsT(elapsed).count(); }
  void PrintElapsed(std::string label) {
    char buffer[255];
    snprintf(buffer, sizeof(buffer), "%s:\t %fs", label.c_str(),
             SecondsT(elapsed).count());
    LOG(CONSOLE) << buffer;
    Reset();
  }
};

/*!
 * \brief Singleton class storing device memory usage statistic. This class is
 * implemented to monitor the device memory usage across XGBoost.
 *
 * Current implementation monitors the pointer passed to cudaMalloc within
 * XGBoost, and the `this` pointer from allocator of thrust::device_vector.
 * Also, it uses backtrace obtained from rdynamic to indicate where did the
 * allocations happen, the backtrace quality varies from platform to platform.
 *
 * Besides, It's possible that the pointer address being reused by system
 * allocator, hence the result from this class is not 100% relible. It's merely
 * providing hints for optimization. Therefore once such a functionality is
 * supported by nvprof or similar tools, we should remove this class and
 * related wrappers.
 */
class DeviceMemoryStat {
  struct Usage {
    std::set<std::string> traces_;
    size_t running_;  // running total.
    size_t peak_;
    size_t count_;  // allocation count.

   public:
    Usage() : running_{0}, peak_{0}, count_{0} {}
    Usage(std::set<std::string> traces,
          size_t running, size_t peak, size_t count);

    Usage& operator+=(Usage const& other);
    Usage& operator-=(Usage const& other);

    size_t GetPeak() const { return peak_; }
    size_t GetAllocCount() const { return count_; }
    size_t GetRunningSum() const { return running_; }
    std::set<std::string> GetTraces() const { return traces_; }
  };

  std::mutex mutex_;
  bool profiling_;

  Usage global_usage_;
  std::map<void const*, Usage> usage_map_;

#if !defined(_MSC_VER)
  std::vector<std::string> const units_;
#else
  std::vector<std::string> units_;
#endif

  DeviceMemoryStat() :
      mutex_(), profiling_{false}, global_usage_()  // NOLINT
#if !defined(_MSC_VER)
      , units_{"B", "KB", "MB", "GB"} {}
#else
  // MSVC 2013: list initialization inside member initializer list or
  // non-static data member initializer is not implemented
  {
    units_.push_back("B");
    units_.push_back("KB");
    units_.push_back("MB");
    units_.push_back("GB");
  }
#endif

 public:
  ~DeviceMemoryStat() { PrintSummary(); }

  /*! \brief Get an instance. */
  static DeviceMemoryStat& Ins() {
    static DeviceMemoryStat instance;
    return instance;
  }

  void SetProfiling(bool profiling) { profiling_ = profiling; }

  void Allocate(void* ptr, size_t size);
  void Deallocate(void* ptr, size_t size);
  void Deallocate(void* ptr);
  /*! \brief replace the usage stat of lhs with the one from rhs. */
  void Replace(void const* lhs, void const* rhs);
  void Reset();

  void PrintSummary() const;

  size_t GetPeakUsage() const;
  size_t GetAllocationCount() const;
  Usage const& GetPtrUsage(void const* ptr) const;
};

/**
 * \struct  Monitor
 *
 * \brief Timing utility used to measure total method execution time over the
 * lifetime of the containing object.
 */
struct Monitor {
 private:
  struct Statistics {
    Timer timer;
    size_t count{0};
    uint64_t nvtx_id;
  };
  std::string label = "";
  std::map<std::string, Statistics> statistics_map;
  Timer self_timer;

 public:
  Monitor() { self_timer.Start(); }

  ~Monitor() {
    if (!ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) return;

    LOG(CONSOLE) << "======== Monitor: " << label << " ========";
    for (auto &kv : statistics_map) {
      if (kv.second.count == 0) {
        LOG(WARNING) <<
            "Timer for " << kv.first << " did not get stopped properly.";
        continue;
      }
      LOG(CONSOLE) << kv.first << ": " << kv.second.timer.ElapsedSeconds()
                   << "s, " << kv.second.count << " calls @ "
                   << std::chrono::duration_cast<std::chrono::microseconds>(
                          kv.second.timer.elapsed / kv.second.count)
                          .count()
                   << "us";
    }
    self_timer.Stop();
  }
  void Init(std::string label) { this->label = label; }
  void Start(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      statistics_map[name].timer.Start();
    }
  }
  void Stop(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      auto &stats = statistics_map[name];
      stats.timer.Stop();
      stats.count++;
    }
  }
  void StartCuda(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      auto &stats = statistics_map[name];
      stats.timer.Start();
#if defined(XGBOOST_USE_NVTX) && defined(__CUDACC__)
      stats.nvtx_id = nvtxRangeStartA(name.c_str());
#endif  // defined(XGBOOST_USE_NVTX) && defined(__CUDACC__)
    }
  }
  void StopCuda(const std::string &name) {
    if (ConsoleLogger::ShouldLog(ConsoleLogger::LV::kDebug)) {
      auto &stats = statistics_map[name];
      stats.timer.Stop();
      stats.count++;
#if defined(XGBOOST_USE_NVTX) && defined(__CUDACC__)
      nvtxRangeEnd(stats.nvtx_id);
#endif  // defined(XGBOOST_USE_NVTX) && defined(__CUDACC__)
    }
  }
};
}  // namespace common
}  // namespace xgboost
