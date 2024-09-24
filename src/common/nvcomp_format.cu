/**
 * Copyright 2024, XGBoost contributors
 */
#include <nvcomp.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/snappy.hpp>

#include "cuda_context.cuh"
#include "nvcomp_format.h"

namespace xgboost::common {
namespace {
enum Algo {
  kLz4,
  kGDefalte,
  kSnappy,
};
}
void DecompCompressedWithManagerFactoryExample(Context const* ctx,
                                               CompressedByteT const* device_input_ptr,
                                               const size_t input_buffer_len) {
  using namespace nvcomp;

  auto stream = ctx->CUDACtx()->Stream();
  const int chunk_size = 1 << 16;
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
  nvcompBatchedGdeflateOpts_t gdeflate_opts{2};
  GdeflateManager gdeflate_mgr{chunk_size, gdeflate_opts, stream};
  // snappy
  nvcompBatchedSnappyOpts_t snappy_opts{};
  SnappyManager snappy_mgr{chunk_size, snappy_opts, stream};
  // // cascaded
  // nvcompBatchedCascadedOpts_t cascaded_opts{chunk_size, data_type};
  // CascadedManager cascaded_mgr{chunk_size, cascaded_opts, stream};

  auto compress = [device_input_ptr, input_buffer_len](auto& mgr) {
    CompressionConfig comp_config = mgr.configure_compression(input_buffer_len);

    std::cout << "max compressed buffer:" << comp_config.max_compressed_buffer_size << std::endl;
    uint8_t* comp_buffer;
    dh::safe_cuda(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

    mgr.compress(device_input_ptr, comp_buffer, comp_config);
    std::size_t comp_size = mgr.get_compressed_output_size(comp_buffer);
    std::cout << "comp size:" << comp_size
              << " compression ratio:" << (static_cast<double>(comp_size) / input_buffer_len)
              << std::endl;
    return comp_buffer;
  };
  Algo algo = kLz4;
  uint8_t* comp_buffer = nullptr;
  switch (algo) {
    case kLz4: {
      comp_buffer = compress(lz4_mgr);
      break;
    }
    case kGDefalte: {
      comp_buffer = compress(gdeflate_mgr);
      break;
    }
    case kSnappy: {
      comp_buffer = compress(snappy_mgr);
      break;
    }
  }

  // Construct a new nvcomp manager from the compressed buffer.
  // Note we could use the nvcomp_manager from above, but here we demonstrate how to create a
  // manager for the use case where a buffer is received and the user doesn't know how it was
  // compressed Also note, creating the manager in this way synchronizes the stream, as the
  // compressed buffer must be read to construct the manager
  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);

  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  uint8_t* res_decomp_buffer;
  dh::safe_cuda(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

  decomp_nvcomp_manager->decompress(res_decomp_buffer, comp_buffer, decomp_config);

  dh::safe_cuda(cudaFree(comp_buffer));
  dh::safe_cuda(cudaFree(res_decomp_buffer));

  stream.Sync();
}
}  // namespace xgboost::common
