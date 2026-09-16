/**
 * Copyright 2026, XGBoost Contributors
 */
#include "cufile_stream.h"

#if defined(__linux__)
#include <cufile.h>
#include <dlfcn.h>   // for dlopen, dlsym, dlclose
#include <unistd.h>  // for pread

#include <algorithm>  // for min
#include <cerrno>     // for errno
#include <cstdio>     // for fopen, fclose, fileno
#include <cstring>    // for strerror
#include <vector>     // for vector

#include "cuda_pinned_allocator.h"  // for PinnedAllocator
#include "io.h"                     // for IOAlignment

namespace xgboost::common {
namespace {
void CheckCuFile(CUfileError_t status) {
  CHECK_EQ(status.err, CU_FILE_SUCCESS)
      << "cuFile: " << CUFILE_ERRSTR(status.err) << ", CUDA error: " << status.cu_err;
}

class CuFileAPI {
  std::unique_ptr<void, int (*)(void*)> library_{dlopen("libcufile.so", RTLD_NOW | RTLD_LOCAL),
                                                 dlclose};

  template <typename T>
  T Load(char const* name) {
    CHECK(library_) << "Failed to load libcufile.so for the GPU disk cache: " << dlerror();
    auto fn = reinterpret_cast<T>(dlsym(library_.get(), name));
    CHECK(fn) << "Failed to load cuFile symbol " << name << ": " << dlerror();
    return fn;
  }

 public:
  decltype(&cuFileHandleRegister) HandleRegister =
      Load<decltype(HandleRegister)>("cuFileHandleRegister");
  decltype(&cuFileHandleDeregister) HandleDeregister =
      Load<decltype(HandleDeregister)>("cuFileHandleDeregister");
  decltype(&cuFileReadAsync) ReadAsync = Load<decltype(ReadAsync)>("cuFileReadAsync");
  decltype(&cuFileStreamRegister) StreamRegister =
      Load<decltype(StreamRegister)>("cuFileStreamRegister");
  decltype(&cuFileStreamDeregister) StreamDeregister =
      Load<decltype(StreamDeregister)>("cuFileStreamDeregister");
  decltype(&cuFileDriverClose) DriverClose = Load<decltype(DriverClose)>("cuFileDriverClose_v2");

  CuFileAPI() { CheckCuFile(Load<decltype(&cuFileDriverOpen)>("cuFileDriverOpen")()); }
  ~CuFileAPI() { DriverClose(); }

  static CuFileAPI& Get() {
    static CuFileAPI api;
    return api;
  }
};
}  // namespace

void InitCuFile() { CuFileAPI::Get(); }

struct CuFileReadStream::Impl {
  CuFileAPI& api{CuFileAPI::Get()};
  std::unique_ptr<std::FILE, decltype(&std::fclose)> file{nullptr, std::fclose};
  std::unique_ptr<void, decltype(&cuFileHandleDeregister)> handle{nullptr, api.HandleDeregister};
  std::size_t offset, remaining;
  // Keep the request and completion status alive until Sync().
  std::size_t size{0};
  off_t file_offset{0}, buffer_offset{0};
  // Give the completion status its own pinned allocation for concurrent reads.
  std::vector<ssize_t, cuda_impl::PinnedAllocator<ssize_t>> bytes_read{0};
  // Give concurrent prefetch workers distinct cuFile streams, instead of the default-stream sentinel.
  curt::Stream stream;
  bool pending{false};

  Impl(StringView path, std::size_t offset, std::size_t length)
      : file{std::fopen(path.c_str(), "rb"), std::fclose}, offset{offset}, remaining{length} {
    CHECK(file) << "Failed to open " << path << ": " << std::strerror(errno);
    CUfileDescr_t desc{};
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    desc.handle.fd = fileno(file.get());
    CUfileHandle_t fh{nullptr};
    CheckCuFile(api.HandleRegister(&fh, &desc));
    handle.reset(fh);
    CheckCuFile(api.StreamRegister(stream.Handle(), CU_FILE_STREAM_FIXED_BUF_OFFSET |
                                                        CU_FILE_STREAM_FIXED_FILE_OFFSET |
                                                        CU_FILE_STREAM_FIXED_FILE_SIZE));
  }
  ~Impl() {
    if (pending) {
      stream.View().Sync(false);
    }
    api.StreamDeregister(stream.Handle());
  }

  void Advance(std::size_t n_bytes) {
    auto n = std::min(remaining, DivRoundUp(n_bytes, IOAlignment()) * IOAlignment());
    offset += n;
    remaining -= n;
  }
};

CuFileReadStream::CuFileReadStream(StringView path, std::size_t offset, std::size_t length)
    : impl_{std::make_unique<Impl>(path, offset, length)} {}
CuFileReadStream::~CuFileReadStream() = default;

bool CuFileReadStream::Read(void* ptr, std::size_t n_bytes) {
  auto& s = *impl_;
  if (n_bytes > s.remaining) {
    return false;
  }
  auto n = pread(fileno(s.file.get()), ptr, n_bytes, s.offset);
  CHECK_GE(n, 0) << "Failed to read ELLPACK cache metadata: " << std::strerror(errno);
  s.Advance(n_bytes);
  return static_cast<std::size_t>(n) == n_bytes;
}

bool CuFileReadStream::ReadAsync(void* ptr, std::size_t n_bytes, curt::StreamRef stream) {
  auto& s = *impl_;
  CHECK(!s.pending);
  if (n_bytes > s.remaining) {
    return false;
  }
  s.size = n_bytes;
  s.file_offset = s.offset;
  s.bytes_read[0] = 0;
  curt::Event ready;
  ready.Record(stream);
  s.stream.Wait(ready);
  CheckCuFile(s.api.ReadAsync(s.handle.get(), ptr, &s.size, &s.file_offset, &s.buffer_offset,
                              s.bytes_read.data(), s.stream.Handle()));
  s.pending = true;
  s.Advance(n_bytes);
  return true;
}

void CuFileReadStream::Sync() {
  auto& s = *impl_;
  if (s.pending) {
    s.stream.Sync();
    s.pending = false;
    CHECK_EQ(s.bytes_read[0], static_cast<ssize_t>(s.size))
        << "Incomplete cuFile read of the ELLPACK cache.";
  }
}
}  // namespace xgboost::common
#else
namespace xgboost::common {
void InitCuFile() { LOG(FATAL) << "cuFile GPU disk cache requires Linux."; }
struct CuFileReadStream::Impl {};
CuFileReadStream::CuFileReadStream(StringView, std::size_t, std::size_t) { InitCuFile(); }
CuFileReadStream::~CuFileReadStream() = default;
bool CuFileReadStream::Read(void*, std::size_t) { return false; }
bool CuFileReadStream::ReadAsync(void*, std::size_t, curt::StreamRef) { return false; }
void CuFileReadStream::Sync() {}
}  // namespace xgboost::common
#endif  // defined(__linux__)
