#pragma once

#include <Eigen/Core>

namespace tmol {
namespace numeric {

unsigned int
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
most_sig_bit128(unsigned int x) {
  // effectively pow(2,int(floor(log2(g.size())))), but avoiding
  // the transcendental functions.
  // The input value, x, must be less than or equal to 128, so
  // it works just fine for numbers that are less than or equal
  // to 32 -- the maximum number of active threads in a warp.
  //
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
  return x & ~(x >> 1);
}


}
}
