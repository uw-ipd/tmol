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

  template <int N_T, int WIDTH, typename TD, typename TS>
  static void copy_and_cast_contiguous_data(
      TD* __restrict__ dst, TS* __restrict__ src, int n);

  template <int N_T, typename Func>
  static void for_each_in_workgroup(Func f);

  template <int N_T, typename T, typename S, typename OP>
  static T reduce_in_workgroup(T val, S shared, OP op);

  template <int N_T, typename T, typename S, typename OP>
  static T shuffle_reduce_in_workgroup(T val, OP op);

  static void synchronize_workgroup();
};

}  // namespace common
}  // namespace score
}  // namespace tmol
