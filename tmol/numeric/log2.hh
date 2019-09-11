#pragma once

#include <Eigen/Core>

namespace tmol {
namespace numeric {

// effectively pow(2,int(floor(log2(g.size())))), but avoiding
// transcendental function evaluation. Handy for working with
// coalesced groups, which usually are not an even power of 2
unsigned int EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
most_sig_bit(unsigned int x) {
#ifdef __CUDA_ARCH__
  return 1 << (31 - __clz(x));
#endif
}

}  // namespace numeric
}  // namespace tmol
