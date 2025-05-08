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
#include "cuda_dr_utils.h"
#include "device_vector.cuh"
#include "nvcomp_format.h"

namespace xgboost::common {
namespace {
enum Algo {
  kLz4,
  kGDefalte,
  kSnappy,
};
}

void SafeNvComp(nvcompStatus_t status) {
  if (status != nvcompSuccess) {
    LOG(FATAL) << "NVComp error:" << static_cast<std::int32_t>(status);
  }
}

[[nodiscard]] CuMemParams CompressSnappy(Context const* ctx, Span<CompressedByteT const> in,
                                         dh::DeviceUVector<std::uint8_t>* p_out) {
  auto cuctx = ctx->CUDACtx();

  std::size_t const kChunkSize = 1 << 18;
  auto nvcompBatchedSnappyOpts = nvcompBatchedSnappyDefaultOpts;

  nvcompAlignmentRequirements_t compression_alignment_reqs;
  SafeNvComp(nvcompBatchedSnappyCompressGetRequiredAlignments(nvcompBatchedSnappyOpts,
                                                              &compression_alignment_reqs));
  CHECK_EQ(compression_alignment_reqs.input, 1);
  CHECK_EQ(compression_alignment_reqs.output, 1);
  CHECK_EQ(compression_alignment_reqs.temp, 1);

  /**
   * Inputs
   */
  std::size_t n_chunks = (in.size() + kChunkSize - 1) / kChunkSize;
  std::size_t last = 0;

  std::vector<CompressedByteT const*> h_in_ptrs;
  std::vector<std::size_t> h_in_sizes;
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto n = std::min(kChunkSize, in.size() - last);
    auto chunk = in.subspan(last, n);
    last += n;

    h_in_sizes.push_back(chunk.size());
    h_in_ptrs.push_back(chunk.data());
  }
  CHECK_EQ(last, in.size());

  dh::DeviceUVector<void const*> in_ptrs(h_in_ptrs.size());
  dh::safe_cuda(cudaMemcpyAsync(in_ptrs.data(), h_in_ptrs.data(), Span{h_in_ptrs}.size_bytes(),
                                cudaMemcpyDefault, cuctx->Stream()));
  dh::DeviceUVector<std::size_t> in_sizes(h_in_sizes.size());
  dh::safe_cuda(cudaMemcpyAsync(in_sizes.data(), h_in_sizes.data(), Span{h_in_sizes}.size_bytes(),
                                cudaMemcpyDefault, cuctx->Stream()));

  CHECK_EQ(n_chunks, in_sizes.size());
  std::size_t max_in_nbytes = *std::max_element(h_in_sizes.cbegin(), h_in_sizes.cend());

  /**
   * Outputs
   */
  std::size_t comp_temp_bytes;
  SafeNvComp(nvcompBatchedSnappyCompressGetTempSize(n_chunks, kChunkSize, nvcompBatchedSnappyOpts,
                                                    &comp_temp_bytes));
  CHECK_EQ(comp_temp_bytes, 0);
  dh::DeviceUVector<char> comp_tmp(comp_temp_bytes);

  std::size_t max_out_nbytes;
  SafeNvComp(nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
      std::min(max_in_nbytes, kChunkSize), nvcompBatchedSnappyOpts, &max_out_nbytes));
  p_out->resize(max_out_nbytes * n_chunks);
  std::vector<void*> h_out_ptrs(n_chunks);
  std::vector<std::size_t> h_out_sizes(n_chunks);
  auto s_out = dh::ToSpan(*p_out);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto chunk = s_out.subspan(max_out_nbytes * i, max_out_nbytes);
    h_out_ptrs[i] = chunk.data();
    h_out_sizes[i] = chunk.size();
  }
  dh::DeviceUVector<void*> out_ptrs(h_out_ptrs.size());
  dh::safe_cuda(cudaMemcpyAsync(out_ptrs.data(), h_out_ptrs.data(), Span{h_out_ptrs}.size_bytes(),
                                cudaMemcpyDefault));
  dh::DeviceUVector<std::size_t> out_sizes(h_out_sizes.size());
  dh::safe_cuda(cudaMemcpyAsync(out_sizes.data(), h_out_sizes.data(),
                                Span{h_out_sizes}.size_bytes(), cudaMemcpyDefault));

  /**
   * Compress
   */
  SafeNvComp(nvcompBatchedSnappyCompressAsync(
      in_ptrs.data(), in_sizes.data(), max_in_nbytes, n_chunks, comp_tmp.data(), comp_temp_bytes,
      out_ptrs.data(), out_sizes.data(), nvcompBatchedSnappyOpts, cuctx->Stream()));
  auto n_bytes = thrust::reduce(cuctx->CTP(), out_sizes.cbegin(), out_sizes.cend());
  auto n_total_bytes = p_out->size();
  auto ratio = static_cast<double>(n_total_bytes) / in.size_bytes();
  LOG(INFO) << "Input: " << in.size_bytes() << ", need:" << n_bytes
            << " allocated:" << n_total_bytes << " ratio:" << ratio;

  /**
   * Meta
   */
  CuMemParams params(n_chunks);
  dh::safe_cuda(cudaMemcpyAsync(h_out_sizes.data(), out_sizes.data(),
                                Span{h_out_sizes}.size_bytes(), cudaMemcpyDefault,
                                cuctx->Stream()));
  std::memset(params.data(), '\0', params.size() * sizeof(CuMemParams::value_type));
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto& p = params[i];
    p.srcNumBytes = h_out_sizes[i];
    p.dstNumBytes = h_in_sizes[i];
    p.algo = CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY;
  }
  return params;
}

void DecompressSnappy(dh::CUDAStreamView s, CuMemParams params, Span<CompressedByteT const> in,
                      Span<CompressedByteT> out) {

  std::size_t error_index = 0;
  std::size_t n_chunks = params.size();

  std::vector<cuuint32_t, cuda_impl::PinnedPoolAllocator<cuuint32_t>> act_bytes(n_chunks, 0);
  std::size_t last_in = 0, last_out = 0;

  std::vector<void const*> in_chunk_ptrs(n_chunks);
  std::vector<std::size_t> in_chunk_sizes(n_chunks);
  std::vector<std::size_t> out_chunk_sizes(n_chunks);
  // std::vector<std::size_t> act(n_chunks, 0);
  std::vector<void*> out_ptrs(n_chunks);
  dh::device_vector<nvcompStatus_t> status(n_chunks);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    in_chunk_ptrs[i] = in.subspan(last_in, params[i].srcNumBytes).data();
    in_chunk_sizes[i] = params[i].srcNumBytes;
    out_chunk_sizes[i] = params[i].dstNumBytes;
    out_ptrs[i] = out.subspan(last_out, params[i].dstNumBytes).data();

    last_in += params[i].srcNumBytes;
    last_out += params[i].dstNumBytes;
  }
  // copy to d
  dh::device_vector<void const*> d_in_chunk_ptrs(in_chunk_ptrs);
  dh::device_vector<std::size_t> d_in_chunk_sizes(in_chunk_sizes);
  dh::device_vector<std::size_t> d_out_chunk_sizes(out_chunk_sizes);
  dh::device_vector<std::size_t> act(n_chunks, 0);
  dh::device_vector<void*> d_out_ptrs(out_ptrs);
  SafeNvComp(nvcompBatchedSnappyDecompressAsync(
      d_in_chunk_ptrs.data().get(), d_in_chunk_sizes.data().get(), d_out_chunk_sizes.data().get(),
      act.data().get(), n_chunks, nullptr, 0, d_out_ptrs.data().get(), status.data().get(), s));
  s.Sync();

  // for (std::size_t i = 0; i < n_chunks; ++i) {
  //   params[i].dstActBytes = act_bytes.data() + i;
  //   params[i].src = in.subspan(last_in, params[i].srcNumBytes).data();
  //   params[i].dst = out.subspan(last_out, params[i].dstNumBytes).data();
  //   last_in += params[i].srcNumBytes;
  //   last_out += params[i].dstNumBytes;
  // }
  //
  // auto err = cudr::GetGlobalCuDriverApi().cuMemBatchDecompressAsync(params.data(), n_chunks,
  //                                                                   0 /*unused*/, &error_index, s);
  // safe_cu(err);
}

void CompressEllpack(Context const* ctx, Span<CompressedByteT const> in,
                     dh::DeviceUVector<std::uint8_t>* p_out) {
  auto stream = ctx->CUDACtx()->Stream();
  const int chunk_size = 1 << 16;
  // NVCOMP_TYPE_UINT8
  nvcompType_t data_type = NVCOMP_TYPE_UCHAR;

  // lz4
  nvcompBatchedLZ4Opts_t lz4_opts{data_type};
  nvcomp::LZ4Manager lz4_mgr{chunk_size, lz4_opts, stream};
  // gdeflate
  /**
   * 0 : high-throughput, low compression ratio (default)
   * 1 : low-throughput, high compression ratio
   * 2 : highest-throughput, entropy-only compression (use for symmetric compression/decompression
   * performance)
   */
  nvcompBatchedGdeflateOpts_t gdeflate_opts{1};
  nvcomp::GdeflateManager gdeflate_mgr{chunk_size, gdeflate_opts, stream};
  // snappy
  nvcompBatchedSnappyOpts_t snappy_opts{};
  nvcomp::SnappyManager snappy_mgr{chunk_size, snappy_opts, stream};
  // cascaded
  nvcompBatchedCascadedOpts_t cascaded_opts{chunk_size, data_type};
  nvcomp::CascadedManager cascaded_mgr{chunk_size, cascaded_opts, stream};
  dh::DeviceUVector<std::uint8_t>& comp_buffer = *p_out;

  auto compress = [in, &comp_buffer](auto& mgr) {
    // This may fail with:
    // Could not determine the maximum compressed chunk size. : code=11.
    nvcomp::CompressionConfig comp_config = mgr.configure_compression(in.size_bytes());
    mgr.set_scratch_allocators(
        [](std::size_t n_bytes) -> void* {
          return static_cast<void*>(dh::XGBDeviceAllocator<char>{}.allocate(n_bytes).get());
        },
        [](void* ptr, std::size_t n_bytes) {
          dh::XGBDeviceAllocator<char>{}.deallocate(
              thrust::device_ptr<char>{static_cast<char*>(ptr)}, n_bytes);
        });

    comp_buffer.resize(comp_config.max_compressed_buffer_size);

    mgr.compress(in.data(), comp_buffer.data(), comp_config);
    std::size_t comp_size = mgr.get_compressed_output_size(comp_buffer.data());
    LOG(INFO) << "max compressed buffer:"
              << common::HumanMemUnit(comp_config.max_compressed_buffer_size)
              << " compression size:" << common::HumanMemUnit(comp_size)
              << " compression ratio:" << (static_cast<double>(comp_size) / in.size_bytes())
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

void DecompressEllpack(curt::CUDAStreamView s, Span<CompressedByteT const> in,
                       Span<CompressedByteT> out) {
  auto decomp_nvcomp_manager = nvcomp::create_manager(in.data(), s);
  decomp_nvcomp_manager->set_scratch_allocators(
      [](std::size_t n_bytes) -> void* {
        return static_cast<void*>(dh::XGBDeviceAllocator<char>{}.allocate(n_bytes).get());
      },
      [](void* ptr, std::size_t n_bytes) {
        dh::XGBDeviceAllocator<char>{}.deallocate(thrust::device_ptr<char>{static_cast<char*>(ptr)},
                                                  n_bytes);
      });
  nvcomp::DecompressionConfig decomp_config =
      decomp_nvcomp_manager->configure_decompression(in.data());
  CHECK_GE(out.size_bytes(), decomp_config.decomp_data_size);
  decomp_nvcomp_manager->decompress(out.data(), in.data(), decomp_config);
}
}  // namespace xgboost::common
