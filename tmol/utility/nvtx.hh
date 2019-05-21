#pragma once

#if false && defined(WITH_CUDA) && defined(WITH_NVTX)

#include <nvToolsExt.h>

#define nvtx_range_push(n) nvtxRangePushA(n)
#define nvtx_range_pop() nvtxRangePop()
#define nvtx_range_function() \
  NVTXRange __nvtx_range_function__(__PRETTY_FUNCTION__)

#else

#define nvtx_range_push(n)
#define nvtx_range_pop()
#define nvtx_range_function()

#endif

struct NVTXRange {
  bool enabled = false;

  NVTXRange(const char* name) {
    nvtx_range_push(name);
    enabled = true;
  }

  void exit() {
    if (enabled) {
      nvtx_range_pop();
      enabled = false;
    }
  }

  ~NVTXRange() { exit(); }
};
