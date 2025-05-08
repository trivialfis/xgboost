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

std::size_t CalcInChunks(dh::CUDAStreamView stream, Span<CompressedByteT const> data,
                         std::size_t chunk_size, dh::DeviceUVector<void const*>* p_ptrs,
                         dh::DeviceUVector<std::size_t>* p_sizes, std::size_t* max_n_bytes) {
  // div roundup
  std::size_t n_chunks = (data.size() + chunk_size - 1) / chunk_size;
  std::size_t last = 0;

  std::vector<CompressedByteT const*> ptrs;
  std::vector<std::size_t> sizes;
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto n = std::min(chunk_size, data.size() - last);
    auto chunk = data.subspan(last, n);
    last += n;

    sizes.push_back(chunk.size());
    ptrs.push_back(chunk.data());
  }
  CHECK_EQ(last, data.size());

  p_sizes->resize(sizes.size());
  dh::safe_cuda(cudaMemcpyAsync(p_sizes->data(), sizes.data(), Span{sizes}.size_bytes(),
                                cudaMemcpyDefault, stream));

  p_ptrs->resize(ptrs.size());
  dh::safe_cuda(cudaMemcpyAsync(p_ptrs->data(), ptrs.data(), Span{ptrs}.size_bytes(),
                                cudaMemcpyDefault, stream));

  CHECK_EQ(n_chunks, p_sizes->size());
  *max_n_bytes = *std::max_element(sizes.cbegin(), sizes.cend());
  return n_chunks;
}

void CompressSnappy(Context const* ctx, Span<CompressedByteT const> in,
                    dh::DeviceUVector<std::uint8_t>* p_out) {
  std::size_t const kChunkSize = 1 << 18;  // fixme: this might be too large for memory alloc
  auto nvcompBatchedSnappyOpts = nvcompBatchedSnappyDefaultOpts;

  nvcompAlignmentRequirements_t compression_alignment_reqs;
  SafeNvComp(nvcompBatchedSnappyCompressGetRequiredAlignments(nvcompBatchedSnappyOpts,
                                                              &compression_alignment_reqs));
  CHECK_EQ(compression_alignment_reqs.input, 1);
  CHECK_EQ(compression_alignment_reqs.output, 1);
  CHECK_EQ(compression_alignment_reqs.temp, 1);

  // inputs
  dh::DeviceUVector<void const*> in_ptrs;
  dh::DeviceUVector<std::size_t> in_sizes;

  std::size_t comp_temp_bytes;
  auto cuctx = ctx->CUDACtx();
  std::size_t max_n_bytes = 0;
  auto n_chunks = CalcInChunks(cuctx->Stream(), in, kChunkSize, &in_ptrs, &in_sizes, &max_n_bytes);
  SafeNvComp(nvcompBatchedSnappyCompressGetTempSize(n_chunks, kChunkSize, nvcompBatchedSnappyOpts,
                                                    &comp_temp_bytes));
  CHECK_EQ(comp_temp_bytes, 0);
  dh::DeviceUVector<char> comp_tmp(comp_temp_bytes);

  std::size_t max_out_bytes;
  SafeNvComp(nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
      std::min(max_n_bytes, kChunkSize), nvcompBatchedSnappyOpts, &max_out_bytes));
  p_out->resize(max_out_bytes * n_chunks);
  std::vector<void*> h_out_ptrs(n_chunks);
  std::vector<std::size_t> h_out_sizes(n_chunks);
  auto s_out = dh::ToSpan(*p_out);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto chunk = s_out.subspan(max_out_bytes * i, max_out_bytes);
    h_out_ptrs[i] = chunk.data();
    h_out_sizes[i] = chunk.size();
  }
  dh::DeviceUVector<void*> out_ptrs(h_out_ptrs.size());
  dh::safe_cuda(cudaMemcpyAsync(out_ptrs.data(), h_out_ptrs.data(), Span{h_out_ptrs}.size_bytes(),
                                cudaMemcpyDefault));
  dh::DeviceUVector<std::size_t> out_sizes(h_out_sizes.size());
  dh::safe_cuda(cudaMemcpyAsync(out_sizes.data(), h_out_sizes.data(),
                                Span{h_out_sizes}.size_bytes(), cudaMemcpyDefault));
  // build output buffers
  SafeNvComp(nvcompBatchedSnappyCompressAsync(
      in_ptrs.data(), in_sizes.data(), max_n_bytes, n_chunks, comp_tmp.data(), comp_temp_bytes,
      out_ptrs.data(), out_sizes.data(), nvcompBatchedSnappyOpts, cuctx->Stream()));
  auto n_bytes = thrust::reduce(cuctx->CTP(), out_sizes.cbegin(), out_sizes.cend());
  auto n_total_bytes = p_out->size();
  auto ratio = static_cast<double>(n_total_bytes) / in.size_bytes();
  LOG(INFO) << "Input: " << in.size_bytes() << ", need:" << n_bytes
            << " allocated:" << n_total_bytes << " ratio:" << ratio;
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
  std::size_t error_index = 0;
  std::size_t num_chunks = 1;
  CUmemDecompressParams params;
  std::memset(&params, '\0', sizeof(params));
  params.srcNumBytes = in.size_bytes();
  params.dstNumBytes = out.size_bytes();
  cuuint32_t dst_act_bytes = 0;
  params.dstActBytes = &dst_act_bytes;
  params.src = in.data();
  params.dst = out.data();
  params.algo = CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY;
  auto err = cudr::GetGlobalCuDriverApi().cuMemBatchDecompressAsync(&params, num_chunks,
                                                                    1 /*unused*/, &error_index, s);
  safe_cu(err);

  auto decomp_nvcomp_manager = nvcomp::create_manager(in.data(), s);
  // decomp_nvcomp_manager->set_scratch_allocators(
  //     [](std::size_t n_bytes) -> void* {
  //       return static_cast<void*>(dh::XGBDeviceAllocator<char>{}.allocate(n_bytes).get());
  //     },
  //     [](void* ptr, std::size_t n_bytes) {
  //       dh::XGBDeviceAllocator<char>{}.deallocate(thrust::device_ptr<char>{static_cast<char*>(ptr)},
  //                                                 n_bytes);
  //     });
  // nvcomp::DecompressionConfig decomp_config =
  //     decomp_nvcomp_manager->configure_decompression(in.data());
  // CHECK_GE(out.size_bytes(), decomp_config.decomp_data_size);
  // decomp_nvcomp_manager->decompress(out.data(), comp_buffer, decomp_config);
}
}  // namespace xgboost::common
