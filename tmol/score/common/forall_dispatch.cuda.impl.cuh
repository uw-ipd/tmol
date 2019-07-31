#pragma once

#include <moderngpu/transform.hxx>

#include "forall_dispatch.hh"

namespace tmol {
namespace score {
namespace common {

template <>
struct ForallDispatch<tmol::Device::CUDA> {
  template <typename Int, typename Func>
  static void forall(Int N, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform(f, N, context);
  }

  template <typename Int, typename Func>
  static void forall_stacks(Int Nstacks, Int N, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int stack = index / N;
          int i = index % N;
          f(stack, i);
        },
        N * Nstacks,
        context);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
