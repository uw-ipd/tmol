
#pragma once

#include "device_operations.hh"

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::CPU> {
  template <typename launch_t, typename Func>
  static void forall(int N, Func f) {
    for (int i = 0; i < N; ++i) {
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

  template <int TILE_SIZE, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f) {
    for (int i = 0; i < n_workgroups; ++i) {
      f(i);
    }
  }

  template <int TILE_SIZE, int WIDTH, typename T>
  static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n) {
    for (int i = 0; i < n; ++i) {
      dst[i] = src[i];
    }
  }

  template <int TILE_SIZE, typename Func>
  static void for_each_in_workgroup(Func f) {
    for (int i = 0; i < TILE_SIZE; ++i) {
      f(i);
    }
  }

  // template <int TILE_SIZE, int WIDTH>
  // static void copy_contiguous_float_data(float * __restrict__ dst, float *
  // __restrict__ src, int n)
  // {
  //   for (int i = 0; i < n; ++i) {
  //     dst[i] = src[i];
  //   }
  // }

  // template <typename T, int TILE_SIZE, int WIDTH>
  // static void copy_contingent(int n, Func f) {
  //   for (int i = 0; i < n; ++i) {
  //     f(i);
  //   }
  // }

  // No op on 1-core CPU
  template <int TILE_SIZE, typename T, typename S, typename OP>
  static T reduce_in_workgroup(T val, S, OP) {
    return val;
  }

  template <int N_T, typename T, typename S, typename OP>
  static T shuffle_reduce_in_workgroup(T val, OP) {
    return val;
  }

  // No op on 1-core CPU
  static void synchronize_workgroup() {}
};

}  // namespace common
}  // namespace score
}  // namespace tmol
