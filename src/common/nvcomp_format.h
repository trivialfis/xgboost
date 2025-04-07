/**
 * Copyright 2025, XGBoost contributors
 */
#include "compressed_iterator.h"
#include "xgboost/context.h"  // for Context

namespace xgboost::common {
void DecompCompressedWithManagerFactoryExample(Context const* ctx,
                                               CompressedByteT const* device_input_ptrs,
                                               const size_t input_buffer_len);

void CompressEllpack(Context const* ctx, CompressedByteT const* device_input_ptr,
                     std::size_t input_buffer_len, dh::DeviceUVector<std::uint8_t>* p_out);
}  // namespace xgboost::common
