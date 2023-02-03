#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct DeviceOperations {
  template <typename launch_t, typename Func>
  static void forall(int N, Func f);

  template <typename Int, typename Func>
  static void forall_stacks(Int Nstacks, Int N, Func f);

  template <typename Int, typename Func>
  static void foreach_combination_triple(Int dim1, Int dim2, Int dim3, Func f);

  template <typename launch_t, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f);

  template <int N_T, int WIDTH, typename T>
  static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n);

  // Copy a block of contiguous data from src to dst and cast it
  // to type TD from type TS
  template <int N_T, int WIDTH, typename TD, typename TS>
  static void copy_contiguous_dat_aand_cast(
      TD* __restrict__ dst, TS* __restrict__ src, int n);

  template <int N_T, typename Func>
  static void for_each_in_workgroup(Func f);

  template <int N_T, typename T, typename S, typename OP>
  static T reduce_in_workgroup(T val, S shared, OP op);

  template <int N_T, typename T, typename S, typename OP>
  static T shuffle_reduce_in_workgroup(T val, OP op);

  static void synchronize_workgroup();

  // Perform an in-place exclusive scan within the workgroup
  // over an array of WIDTH elements. If
  // broadcast_and_return_total is true, then all members
  // of the workgroup will receive the total of the scan
  template <int N_T, int WIDTH, typename T, typename OP>
  static T exclusive_scan_in_workgroup(
      T* data, T identity, OP op, bool broadcast_and_return_total);

  // Perform an in-place inclusive scan within the workgroup
  // over an array of WIDTH elements. If
  // broadcast_and_return_total is true, then all members
  // of the workgroup will receive the total of the scan
  template <int N_T, int WIDTH, typename T, typename OP>
  static T inclusive_scan_in_workgroup(
      T* data, T identity, OP op, bool broadcast_and_return_total);

  // Perform an in-place inclusive segmented scan within the
  // workgroup over an array of WIDTH elements with the
  // seg_begin array informing where each segment begins.
  template <int N_T, int WIDTH, typename T, typename S, typename OP>
  static void inclusive_seg_scan_in_workgroup(
      T* data, T identity, S* seg_begin, OP op);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
