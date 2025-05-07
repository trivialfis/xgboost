/**
 * Copyright 2024-2025, XGBoost contributors
 */
#include <nvcomp.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/snappy.hpp>

#include "cuda_context.cuh"
#include "device_vector.cuh"
#include "nvcomp_format.h"
#include "ref_resource_view.cuh"
#include "ref_resource_view.h"

namespace xgboost::common {
namespace {
enum Algo {
  kLz4,
  kGDefalte,
  kSnappy,
};
}
// fixme: span
void CompressEllpack(Context const* ctx, CompressedByteT const* device_input_ptr,
                     std::size_t input_buffer_len, dh::DeviceUVector<std::uint8_t>* p_out) {
  using namespace nvcomp;

  auto stream = ctx->CUDACtx()->Stream();
  const int chunk_size = 1 << 16;
  // NVCOMP_TYPE_UINT8
  nvcompType_t data_type = NVCOMP_TYPE_UCHAR;

  // lz4
  nvcompBatchedLZ4Opts_t lz4_opts{data_type};
  LZ4Manager lz4_mgr{chunk_size, lz4_opts, stream};
  // gdeflate
  /**
   * 0 : high-throughput, low compression ratio (default)
   * 1 : low-throughput, high compression ratio
   * 2 : highest-throughput, entropy-only compression (use for symmetric compression/decompression
   * performance)
   */
  nvcompBatchedGdeflateOpts_t gdeflate_opts{1};
  GdeflateManager gdeflate_mgr{chunk_size, gdeflate_opts, stream};
  // snappy
  nvcompBatchedSnappyOpts_t snappy_opts{};
  SnappyManager snappy_mgr{chunk_size, snappy_opts, stream};
  // cascaded
  nvcompBatchedCascadedOpts_t cascaded_opts{chunk_size, data_type};
  CascadedManager cascaded_mgr{chunk_size, cascaded_opts, stream};
  dh::DeviceUVector<std::uint8_t>& comp_buffer = *p_out;

  auto compress = [device_input_ptr, input_buffer_len, &comp_buffer](auto& mgr) {
    // This may fail with:
    // Could not determine the maximum compressed chunk size. : code=11.
    CompressionConfig comp_config = mgr.configure_compression(input_buffer_len);
    mgr.set_scratch_allocators(
        [](std::size_t n_bytes) -> void* {
          return static_cast<void*>(dh::XGBDeviceAllocator<char>{}.allocate(n_bytes).get());
        },
        [](void* ptr, std::size_t n_bytes) {
          dh::XGBDeviceAllocator<char>{}.deallocate(
              thrust::device_ptr<char>{static_cast<char*>(ptr)}, n_bytes);
        });


    comp_buffer.resize(comp_config.max_compressed_buffer_size);

    mgr.compress(device_input_ptr, comp_buffer.data(), comp_config);
    std::size_t comp_size = mgr.get_compressed_output_size(comp_buffer.data());
    LOG(INFO) << "max compressed buffer:"
              << common::HumanMemUnit(comp_config.max_compressed_buffer_size)
              << " compression size:" << common::HumanMemUnit(comp_size)
              << " compression ratio:" << (static_cast<double>(comp_size) / input_buffer_len)
              << std::endl;
    comp_buffer.resize(comp_size);
  };
  Algo algo = kSnappy;
  switch (algo) {
    case kLz4: {
      compress(lz4_mgr);
      break;
    }
    case kGDefalte: {
      compress(gdeflate_mgr);
      break;
    }
    case kSnappy: {
      compress(snappy_mgr);
      break;
    }
  }
}

void DecompressEllpack(curt::CUDAStreamView s, CompressedByteT const* comp_buffer,
                       CompressedByteT* out, std::size_t out_n_bytes) {
  auto decomp_nvcomp_manager = nvcomp::create_manager(comp_buffer, s);
  decomp_nvcomp_manager->set_scratch_allocators(
      [](std::size_t n_bytes) -> void* {
        return static_cast<void*>(dh::XGBDeviceAllocator<char>{}.allocate(n_bytes).get());
      },
      [](void* ptr, std::size_t n_bytes) {
        dh::XGBDeviceAllocator<char>{}.deallocate(thrust::device_ptr<char>{static_cast<char*>(ptr)},
                                                  n_bytes);
      });

  nvcomp::DecompressionConfig decomp_config =
      decomp_nvcomp_manager->configure_decompression(comp_buffer);
  CHECK_GE(out_n_bytes, decomp_config.decomp_data_size);
  // auto decomp_buf =
  //     common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(decomp_config.decomp_data_size);
  decomp_nvcomp_manager->decompress(out, comp_buffer, decomp_config);
}
}  // namespace xgboost::common
