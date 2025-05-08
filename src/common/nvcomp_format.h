/**
 * Copyright 2025, XGBoost contributors
 */
#include "compressed_iterator.h"  // for CompressedByteT
#include "cuda_rt_utils.h"        // for CUDAStreamView
#include "xgboost/context.h"      // for Context

namespace xgboost::common {
void CompressSnappy(Context const* ctx, Span<CompressedByteT const> in,
                    dh::DeviceUVector<std::uint8_t>* p_out);

void CompressEllpack(Context const* ctx, Span<CompressedByteT const> in,
                     dh::DeviceUVector<std::uint8_t>* p_out);

void DecompressEllpack(curt::CUDAStreamView s, Span<CompressedByteT const> in,
                       Span<CompressedByteT> out);
}  // namespace xgboost::common
