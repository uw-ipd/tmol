#pragma once

#ifdef __NVCC__
error_this_should_not_be_compiled();  // nvcc should not include this file
#endif

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

  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static void scan(T* src, T* dst, int n, OP op) {
    T last_val = src[0];
    if (scan_type == mgpu::scan_type_inc) {
      dst[0] = last_val;
    }
    for (int i = 1; i < n; ++i) {
      T i_val = src[i];
      T next_val = op(last_val, i_val);
      dst[i] = (scan_type == mgpu::scan_type_exc) ? last_val : next_val;
      last_val = next_val;
    }
  }

  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static T scan_and_return_total(T* src, T* dst, int n, OP op) {
    T last_val = src[0];
    if (scan_type == mgpu::scan_type_inc) {
      dst[0] = last_val;
    }
    for (int i = 1; i < n; ++i) {
      T i_val = src[i];
      T next_val = op(last_val, i_val);
      dst[i] = (scan_type == mgpu::scan_type_exc) ? last_val : next_val;
      printf("scan %d: %d\n", i, dst[i]);
      last_val = next_val;
    }
    return last_val;
  }

  // Segmented scan expects the indices for the beginning of each segment rather
  // than, e.g., a boolean tensor indicating the start of each segment.
  // The identity value (e.g. 0) must be given because pre-initialization is not
  // always possible. seg_starts_inds must be sorted in ascending order.
  template <mgpu::scan_type_t scan_type, typename T, typename Int, typename OP>
  static auto segmented_scan(
      T* src, Int* seg_start_inds, int n, int n_segs, OP op, T identity)
      -> TPack<T, 1, D>;
  {
    auto dst_t = TPack<T, 1, D>::empty({n});
    auto dst = dst_t.view;
    T last_val = identity;  // position 0 is always the start of a segment
    int count_seg = 0;
    for (int i = 0; i < n; ++i) {
      T i_val = src[i];
      if (i == seg_start_inds[count_seg]) {
        last_val = identity;
        count_seg++;
      }
      T next_val = op(last_val, i_val);
      dst[i] = (scan_type == mgpu::scan_type_exc) ? last_val : next_val;
      last_val = next_val;
    }
    return dst_t;
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
};

}  // namespace common
}  // namespace score
}  // namespace tmol
