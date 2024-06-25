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
    mgpu::standard_context_t context(false);
    mgpu::transform(f, N, context);
  }

  template <typename Int, typename Func>
  static void forall_stacks(Int Nstacks, Int N, Func f) {
    mgpu::standard_context_t context(false);
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int stack = index / N;
          int i = index % N;
          f(stack, i);
        },
        N * Nstacks,
        context);
  }

  template <typename Int, typename Func>
  static void foreach_combination_triple(Int dim1, Int dim2, Int dim3, Func f) {
    mgpu::standard_context_t context(false);
    mgpu::transform(
        [=] MGPU_DEVICE(int index) {
          int i = index / (dim2 * dim3);
          index = index % (dim2 * dim3);
          int j = index / dim3;
          int k = index % dim3;
          f(i, j, k);
        },
        dim1 * dim2 * dim3,
        context);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
