/**
 * Copyright 2024, XGBoost contributors
 */
#include "xgboost/context.h"  // for Context
#include "compressed_iterator.h"

namespace xgboost::common {
void DecompCompressedWithManagerFactoryExample(Context const* ctx,
                                               CompressedByteT const* device_input_ptrs,
                                               const size_t input_buffer_len);
}
