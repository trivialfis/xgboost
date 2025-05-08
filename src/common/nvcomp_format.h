/**
 * Copyright 2025, XGBoost contributors
 */
#include <cuda.h>  // for CUmemDecompressParams

#include "compressed_iterator.h"  // for CompressedByteT
#include "cuda_pinned_allocator.h"
#include "cuda_rt_utils.h"    // for CUDAStreamView
#include "xgboost/context.h"  // for Context

namespace xgboost::common {
using CuMemParams =
    std::vector<CUmemDecompressParams, cuda_impl::PinnedPoolAllocator<CUmemDecompressParams>>;

[[nodiscard]] CuMemParams CompressSnappy(Context const* ctx, Span<CompressedByteT const> in,
                                         dh::DeviceUVector<std::uint8_t>* p_out);

void DecompressSnappy(dh::CUDAStreamView s, CuMemParams params, Span<CompressedByteT const> in,
                      Span<CompressedByteT> out);

void CompressEllpack(Context const* ctx, Span<CompressedByteT const> in,
                     dh::DeviceUVector<std::uint8_t>* p_out);

void DecompressEllpack(curt::CUDAStreamView s, Span<CompressedByteT const> in,
                       Span<CompressedByteT> out);
}  // namespace xgboost::common
