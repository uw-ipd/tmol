
#pragma once

#include "forall_dispatch.hh"

namespace tmol {
namespace score {
namespace common {

template <>
struct ForallDispatch<tmol::Device::CPU> {
  template <typename Int, typename Func>
  static void forall(Int N, Func f) {
    for (Int i = 0; i < N; ++i) {
      f(i);
    }
  }

  template <typename Int, typename Func>
  static void forall_stacks(Int Nstacks, Int N, Func f) {
    for (int stack = 0; stack < Nstacks; ++stack) {
      for (Int i = 0; i < N; ++i) {
        f(stack, i);
      }
    }
  }

  template <typename Int, typename Func>
  static void foreach_combination_triple(Int dim1, Int dim2, Int dim3, Func f) {
    for (Int i = 0; i < dim1; ++i) {
      for (Int j = 0; j < dim2; ++j) {
        for (Int k = 0; k < dim3; ++k) {
          f(i, j, k);
        }
      }
    }
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
