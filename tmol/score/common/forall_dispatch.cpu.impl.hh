
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
    for (int stack = 0; stack < NStacks; ++stack) {
      for (Int i = 0; i < N; ++i) {
        f(stack, i);
      }
    }
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
