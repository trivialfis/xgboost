/**
 * Copyright 2026, XGBoost Contributors
 */
#include "cufile_stream.h"

#if defined(__linux__)
#include <cufile.h>
#include <dlfcn.h>  // for dlopen, dlsym, dlclose

#include <cerrno>   // for errno
#include <cstdio>   // for fopen, fclose, fileno
#include <cstring>  // for strerror
#include <mutex>    // for lock_guard, mutex
#include <utility>  // for move
#include <vector>   // for vector

#include "cuda_pinned_allocator.h"  // for PinnedAllocator

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
  decltype(&cuFileWriteAsync) WriteAsync = Load<decltype(WriteAsync)>("cuFileWriteAsync");
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

struct CuFileStream::Impl {
  struct File {
    CuFileAPI& api{CuFileAPI::Get()};
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file{nullptr, std::fclose};
    std::unique_ptr<void, decltype(&cuFileHandleDeregister)> handle{nullptr, api.HandleDeregister};
    curt::Stream stream;
    std::mutex submit_mutex;

    File(StringView path, bool write)
        : file{std::fopen(path.c_str(), write ? "w+b" : "rb"), std::fclose} {
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
    ~File() { api.StreamDeregister(stream.Handle()); }
  };
  std::shared_ptr<File> file;
  // Keep the request and completion status alive until Sync().
  std::size_t size{0};
  off_t file_offset{0}, buffer_offset{0};
  // Give the completion status its own pinned allocation for concurrent reads.
  std::vector<ssize_t, cuda_impl::PinnedAllocator<ssize_t>> bytes_transferred{0};

  bool pending{false};
  char const* operation{nullptr};

  explicit Impl(std::shared_ptr<File> input) : file{std::move(input)} {}
  ~Impl() {
    if (pending) {
      file->stream.View().Sync(false);
    }
  }

  void Transfer(void* ptr, std::size_t n_bytes, std::size_t offset, curt::StreamRef caller,
                bool write) {
    CHECK(!pending);
    if (n_bytes == 0) {
      return;
    }
    size = n_bytes;
    file_offset = offset;
    bytes_transferred[0] = 0;
    operation = write ? "write" : "read";
    curt::Event ready;
    ready.Record(caller);
    // Keep the cuFile operation's enqueue sequence together on the shared IO stream.
    std::lock_guard guard{file->submit_mutex};
    file->stream.Wait(ready);
    auto transfer = write ? file->api.WriteAsync : file->api.ReadAsync;
    CheckCuFile(transfer(file->handle.get(), ptr, &size, &file_offset, &buffer_offset,
                         bytes_transferred.data(), file->stream.Handle()));
    pending = true;
  }
};

CuFileStream::CuFileStream(StringView path, bool write)
    : impl_{std::make_unique<Impl>(std::make_shared<Impl::File>(path, write))} {}
CuFileStream::CuFileStream(CuFileStream const& other)
    : impl_{std::make_unique<Impl>(other.impl_->file)} {}
CuFileStream::~CuFileStream() = default;

void CuFileStream::ReadAsync(void* ptr, std::size_t n_bytes, std::size_t offset,
                             curt::StreamRef stream) {
  impl_->Transfer(ptr, n_bytes, offset, stream, false);
}

void CuFileStream::WriteAsync(void const* ptr, std::size_t n_bytes, std::size_t offset,
                              curt::StreamRef stream) {
  impl_->Transfer(const_cast<void*>(ptr), n_bytes, offset, stream, true);
}

void CuFileStream::Sync() {
  auto& s = *impl_;
  if (s.pending) {
    s.file->stream.Sync();
    s.pending = false;
    CHECK_EQ(s.bytes_transferred[0], static_cast<ssize_t>(s.size))
        << "Incomplete cuFile " << s.operation << " of the ELLPACK cache.";
  }
}
}  // namespace xgboost::common
#else
namespace xgboost::common {
void InitCuFile() { LOG(FATAL) << "cuFile GPU disk cache requires Linux."; }
struct CuFileStream::Impl {};
CuFileStream::CuFileStream(StringView, bool) { InitCuFile(); }
CuFileStream::CuFileStream(CuFileStream const&) { InitCuFile(); }
CuFileStream::~CuFileStream() = default;
void CuFileStream::ReadAsync(void*, std::size_t, std::size_t, curt::StreamRef) {}
void CuFileStream::WriteAsync(void const*, std::size_t, std::size_t, curt::StreamRef) {}
void CuFileStream::Sync() {}
}  // namespace xgboost::common
#endif  // defined(__linux__)
