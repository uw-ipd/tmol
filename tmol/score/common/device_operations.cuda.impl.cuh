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
    auto wrapper = ([=](int tid, int cta) { f(cta); });
    mgpu::standard_context_t context;
    mgpu::cta_launch<launch_t>(wrapper, n_workgroups, context);
  }

  template <int N_T, int WIDTH, typename T>
  __device__ static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n) {
    mgpu::mem_to_shared<N_T, WIDTH>(src, threadIdx.x, n, dst, false);
  }

  template <int N_T, typename Func>
  __device__ static void for_each_in_workgroup(Func f) {
    f(threadIdx.x);
  }

  template <int N_T, typename T, typename S, typename OP>
  __device__ static T reduce_in_workgroup(T val, S shared, OP op) {
    typedef mgpu::cta_reduce_t<N_T, T> reduce_t;
    return reduce_t().reduce(threadIdx.x, val, shared.reduce, N_T, op);
  }

  // For this to work, NT must be <= 32
  template <int N_T, typename T, typename S, typename OP>
  __device__ static T shuffle_reduce_in_workgroup(T val, S OP op) {
    auto g = cooperative_groups::coalesced_threads();
    T reduced_val =
        tmol::score::common::reduce_tile_shfl(g, local_coords[i], op);
    return reduced_val;
  }

  // For this to work, NT must be <= 32
  template <int N_T, typename T, typename S, typename OP>
  __device__ static T shuffle_reduce_in_workgroup(T val, S OP op) {
    auto g = cooperative_groups::coalesced_threads();
    T reduced_val =
        tmol::score::common::reduce_tile_shfl(g, local_coords[i], op);
    return reduced_val;
  }

  template <int N_T, typename T, typename S, typename OP>
  __device__ static T shuffle_reduce_and_broadcast_in_workgroup(
      T val, S OP op) {
    auto g = cooperative_groups::coalesced_threads();
    for (int i = 0; i < 3; ++i) {
      com[i] = tmol::score::common::reduce_tile_shfl(g, local_coords[i], op);
      com[i] /= n_atoms;
      com[i] = g.shfl(com[i], 0);
    }
  }
  __device__ static void synchronize_workgroup() { __syncthreads(); }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
