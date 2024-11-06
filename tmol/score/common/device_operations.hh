#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/extern/moderngpu/scan_types.hxx>  // CPU-friendly

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

  // Note that dst[0] should be initialized to the identity value (e.g. 0) if
  // scan_type is exclusive.
  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static void scan(T* src, T* dst, int n, OP op);

  // Note that dst[0] should be initialized to the identity value (e.g. 0) if
  // scan_type is exclusive.a
  template <mgpu::scan_type_t scan_type, typename T, typename OP>
  static T scan_and_return_total(T* src, T* dst, int n, OP op);

  // Segmented scan expects the indices for the beginning of each segment rather
  // than, e.g., a boolean tensor indicating the start of each segment.
  // The identity value (e.g. 0) must be given because pre-initialization is not
  // always possible. seg_starts_inds must be sorted in ascending order.
  template <
      mgpu::scan_type_t scan_type,
      typename launch_t,
      typename T,
      typename Int,
      typename OP>
  static auto segmented_scan(
      T* src, Int* seg_start_inds, int n, int n_segs, OP op, T identity)
      -> TPack<T, 1, D>;

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
};

}  // namespace common
}  // namespace score
}  // namespace tmol
