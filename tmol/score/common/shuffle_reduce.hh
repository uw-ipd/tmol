#pragma once

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <moderngpu/operators.hxx>
#include <tmol/numeric/log2.hh>
#endif

namespace tmol {
namespace score {
namespace common {

#ifdef __CUDACC__

// Perform reduction within the active threads.
// F is a functor that needs to be associative and
// commutative
// This function could/should be moved elsewhere.
template <typename T, typename F>
__device__ __inline__ T reduce_tile_shfl(
    cooperative_groups::coalesced_group g, T val, F f) {
  // Adapted from https://devblogs.nvidia.com/cooperative-groups/

  // First: have the lower threads shuffle from the
  // threads that hang off the end of the largest-power-of-2
  // less than or equal to the number of active threads.
  unsigned int const gsize = g.size();
  unsigned int const biggest_pow2_base = numeric::most_sig_bit(gsize);

  unsigned int const overhang = gsize - biggest_pow2_base;
  if (overhang > 0) {
    T const overhang_val = g.shfl_down(val, biggest_pow2_base);
    if (g.thread_rank() < overhang) {
      val = f(val, overhang_val);
    }
  }

  // Second: perform a canonical reduction with the group of
  // active threads; the number of iterations would otherwise
  // have missed the "overhang" set if the first shfl_down
  // above had not been performed.
  for (int i = biggest_pow2_base / 2; i > 0; i /= 2) {
    T const shfl_val = g.shfl_down(val, i);
    val = f(val, shfl_val);
  }

  return val;  // note: only thread 0 will return full sum
}

#endif

}  // namespace common
}  // namespace score
}  // namespace tmol
