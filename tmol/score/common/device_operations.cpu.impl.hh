
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

  template <typename launch_t, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f) {
    for (int i = 0; i < n_workgroups; ++i) {
      f(i);
    }
  }

  template <int N_T, int WIDTH, typename T>
  static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n) {
    for (int i = 0; i < n; ++i) {
      dst[i] = src[i];
    }
  }

  template <int N_T, int WIDTH, typename TD, typename TS>
  static void copy_contiguous_data_and_cast(
      TD* __restrict__ dst, TS* __restrict__ src, int n) {
    for (int i = 0; i < n; ++i) {
      dst[i] = static_cast<TD>(src[i]);
    }
  }

  template <int N_T, typename Func>
  static void for_each_in_workgroup(Func f) {
    for (int i = 0; i < N_T; ++i) {
      f(i);
    }
  }

  // No op on 1-core CPU
  template <int N_T, typename T, typename S, typename OP>
  static T reduce_in_workgroup(T val, S, OP) {
    return val;
  }

  template <int N_T, typename T, typename OP>
  static T shuffle_reduce_in_workgroup(T val, OP) {
    return val;
  }

  // See comments for shuffle_reduce_in_workgroup above
  template <int N_T, typename T, typename OP>
  static T shuffle_reduce_and_broadcast_in_workgroup(T val, OP op) {
    return val;
  }

  // No op on 1-core CPU
  static void synchronize_workgroup() {}

  // Perform an in-place exclusive scan within the workgroup over
  // an array of WIDTH elements. If
  // broadcast_and_return_total is true, then all members
  // of the workgroup will receive the total of the scan
  template <int N_T, int WIDTH, typename T, typename OP>
  static T exclusive_scan_in_workgroup(T* data, T identity, OP op, bool) {
    T last_val = identity;
    T sum = identity;
    for (int i = 0; i < WIDTH; ++i) {
      last_val = data[i];
      data[i] = sum;
      sum = op(sum, last_val);
    }
    return sum;
  }

  // Perform an inclusive scan within the workgroup over
  // an array of WIDTH elements. If
  // broadcast_and_return_total is true, then all members
  // of the workgroup will receive the total of the scan
  template <int N_T, int WIDTH, typename T, typename OP>
  static T inclusive_scan_in_workgroup(
      T* data, T identity, OP op, bool broadcast_and_return_total) {
    T sum = identity;
    for (int i = 0; i < WIDTH; ++i) {
      sum = op(sum, data[i]);
      data[i] = sum;
    }
    return sum;
  }

  // Perform an inclusive segmented scan within the workgroup
  // over an array of WIDTH elements with the seg_begin array
  // informing where each segment begins.
  template <int N_T, int WIDTH, typename T, typename S, typename OP>
  static void inclusive_seg_scan_in_workgroup(
      T* data, T identity, S* seg_begin, OP op) {
    T sum = identity;
    for (int i = 0; i < WIDTH; ++i) {
      if (seg_begin[i]) {
        sum = identity;
      }
      sum = op(sum, data[i]);
      data[i] = sum;
    }
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
