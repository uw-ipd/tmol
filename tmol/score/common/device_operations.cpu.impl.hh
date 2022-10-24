
#pragma once

#include "device_operations.hh"

namespace tmol {
namespace score {
namespace common {

template <>
struct DeviceOperations<tmol::Device::CPU> {
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

  template <int N_T, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f) {
    for (int i = 0; i < n_workgroups; ++i) {
      for (int j = 0; j < N_T; ++j) {
        f(j, i);
      }
    }
  }

  template <int TILE_SIZE, int WIDTH, typename T>
  static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n) {
    for (int i = 0; i < n; ++i) {
      dst[i] = src[i];
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
  static void synchronize_workgroup() {}
};

}  // namespace common
}  // namespace score
}  // namespace tmol
