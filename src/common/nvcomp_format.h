/**
 * Copyright 2025, XGBoost contributors
 */
#include "compressed_iterator.h"  // for CompressedByteT
#include "cuda_rt_utils.h"        // for CUDAStreamView
#include "xgboost/context.h"      // for Context

namespace xgboost::common {
void CompressEllpack(Context const* ctx, CompressedByteT const* device_input_ptr,
                     std::size_t input_buffer_len, dh::DeviceUVector<std::uint8_t>* p_out);

void DecompressEllpack(curt::CUDAStreamView s, CompressedByteT const* comp_buffer,
                       CompressedByteT* out, std::size_t out_n_bytes);
}  // namespace xgboost::common
