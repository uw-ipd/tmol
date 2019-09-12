#pragma once

#include <Eigen/Core>

namespace tmol {
namespace numeric {

// effectively pow(2,int(floor(log2(g.size())))), but avoiding
// transcendental function evaluation. Handy for working with
// coalesced groups, which usually are not an even power of 2
unsigned int EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
most_sig_bit128(unsigned int x) {
#ifdef __CUDA_ARCH__
  return 1 << (31 - __clz(x));
#else

  // Adapted from the "Most Significant 1 Bit" function from:
  //
  // @techreport{magicalgorithms,
  // author={Henry Gordon Dietz},
  // title={{The Aggregate Magic Algorithms}},
  // institution={University of Kentucky},
  // howpublished={Aggregate.Org online technical report},
  // URL={http://aggregate.org/MAGIC/}
  // }
  // Date fetched: 2019/9/2
	
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x & ~(x >> 1);
	
#endif
}

}  // namespace numeric
}  // namespace tmol
