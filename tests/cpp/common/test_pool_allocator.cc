/*!
 * Copyright (c) by XGBoost Contributors 2020
 */
#include <gtest/gtest.h>
#include <xgboost/pool_allocator.h>

namespace xgboost {
TEST(MemoryPoolAllocator, Basic) {
  WithPoolAllocator alloc_0;
  ASSERT_EQ(alloc_0.FrameDepth(), 1);
  auto ptr = alloc_0.Malloc(100);
  {
      WithPoolAllocator alloc_1;
      ASSERT_EQ(alloc_0.FrameDepth(), 2);
      ASSERT_EQ(alloc_0.FrameDepth(), alloc_1.FrameDepth());
      alloc_1.Free(ptr);  // There's only 1 gloabl allocator.
  }
  ASSERT_EQ(alloc_0.FrameDepth(), 1);
}
} // namespace xgboost
