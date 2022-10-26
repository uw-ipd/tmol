#pragma once

#include <moderngpu/transform.hxx>
#include <moderngpu/loadstore.hxx>
#include <moderngpu/cta_reduce.hxx>

#include "device_operations.hh"

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::CUDA> {
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

  template <typename Int, typename Func>
  static void foreach_combination_triple(Int dim1, Int dim2, Int dim3, Func f) {
    mgpu::standard_context_t context;
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

  template <typename launch_t, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f) {
    mgpu::standard_context_t context;
    mgpu::cta_launch<launch_t>(f, n_workgroups, context);
  }

  template <int TILE_SIZE, int WIDTH, typename T>
  __device__ static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n) {
    mgpu::mem_to_shared<TILE_SIZE, WIDTH>(src, threadIdx.x, n, dst, false);
  }

  template <int TILE_SIZE, typename Func>
  __device__ static void for_each_in_workgroup(Func f) {
    f(threadIdx.x);
  }

  __device__ static void synchronize_workgroup() { __syncthreads(); }

  template <int TILE_SIZE, typename T, typename S, typename OP>
  __device__ static T reduce_in_workgroup(T val, S shared, OP op) {
    typedef mgpu::cta_reduce_t<TILE_SIZE, T> reduce_t;
    return reduce_t().reduce(threadIdx.x, val, shared.reduce, TILE_SIZE, op);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
