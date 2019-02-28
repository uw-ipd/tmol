#pragma once

#if defined(WITH_CUDA) && defined(WITH_NVTX)

#include <nvToolsExt.h>

#define nvtx_range_push(n) nvtxRangePushA(n);
#define nvtx_range_pop() nvtxRangePop();

#else

#define nvtx_range_push(n)
#define nvtx_range_pop()

#endif
