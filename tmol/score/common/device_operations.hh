#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct DeviceOperations {
  template <typename Int, typename Func>
  static void forall(Int N, Func f);

  template <typename Int, typename Func>
  static void forall_stacks(Int Nstacks, Int N, Func f);

  template <typename Int, typename Func>
  static void foreach_combination_triple(Int dim1, Int dim2, Int dim3, Func f);

  template <int N_T, typename Func>
  static void foreach_workgroup(int n_workgroups, Func f);

  template <int TILE_SIZE, int WIDTH, typename T>
  static void copy_contiguous_data(
      T* __restrict__ dst, T* __restrict__ src, int n);

  // template <typename T, int TILE_SIZE, int WIDTH>
  // static void copy_contingent(int n, Func f);

  static void synchronize_workgroup();
};

}  // namespace common
}  // namespace score
}  // namespace tmol
