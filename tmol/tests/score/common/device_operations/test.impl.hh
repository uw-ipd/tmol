#pragma once

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/extern/moderngpu/operators.hxx>
#include <tmol/tests/score/common/device_operations/test.hh>
#include <tmol/utility/tensor/context_manager.hh>
#include <Eigen/Core>

namespace tmol {
namespace tests {
namespace score {
namespace common {
namespace device_operations {

#ifdef __NVCC__
using launch_t = mgpu::launch_params_t<128, 2>;
#else
using launch_t = launch_t_cpu<128, 2>;
#endif

template <tmol::Device D>
using DO = tmol::score::common::DeviceOperations<D>;

template <tmol::Device D>
auto DevOpsTests<D>::test_forall(TView<int32_t, 1, D> src)
    -> TPack<int32_t, 1, D> {
  ContextManager mgr;
  int n = src.size(0);
  auto dst_t = TPack<int32_t, 1, D>::empty({n});
  auto dst = dst_t.view;
  DO<D>::template forall<launch_t>(
      mgr, n, [=] EIGEN_DEVICE_FUNC(int i) { dst[i] = src[i] + i; });
  return dst_t;
}

template <tmol::Device D>
auto DevOpsTests<D>::test_forall_stacks(TView<int32_t, 2, D> src)
    -> TPack<int32_t, 2, D> {
  ContextManager mgr;
  int nstacks = src.size(0);
  int n = src.size(1);
  auto dst_t = TPack<int32_t, 2, D>::empty({nstacks, n});
  auto dst = dst_t.view;
  DO<D>::template forall_stacks<int>(
      mgr, nstacks, n, [=] EIGEN_DEVICE_FUNC(int stack, int i) {
        dst[stack][i] = src[stack][i] * 2;
      });
  return dst_t;
}

template <tmol::Device D>
auto DevOpsTests<D>::test_foreach_combination_triple(TView<int32_t, 3, D> src)
    -> TPack<int32_t, 3, D> {
  ContextManager mgr;
  int dim1 = src.size(0);
  int dim2 = src.size(1);
  int dim3 = src.size(2);
  auto dst_t = TPack<int32_t, 3, D>::empty({dim1, dim2, dim3});
  auto dst = dst_t.view;
  DO<D>::template foreach_combination_triple<int>(
      mgr, dim1, dim2, dim3, [=] EIGEN_DEVICE_FUNC(int i, int j, int k) {
        dst[i][j][k] = src[i][j][k] + 1;
      });
  return dst_t;
}

template <tmol::Device D>
auto DevOpsTests<D>::test_foreach_workgroup(TView<int32_t, 1, D> src)
    -> TPack<int32_t, 1, D> {
  ContextManager mgr;
  int n = src.size(0);
  auto dst_t = TPack<int32_t, 1, D>::empty({n});
  auto dst = dst_t.view;
  DO<D>::template foreach_workgroup<launch_t>(
      mgr, n, [=] EIGEN_DEVICE_FUNC(int wg) { dst[wg] = src[wg] + wg; });
  return dst_t;
}

template <tmol::Device D>
auto DevOpsTests<D>::test_foreach_pose_workgroup(TView<int32_t, 2, D> src)
    -> TPack<int32_t, 2, D> {
  ContextManager mgr;
  int const n_poses = src.size(0);
  int const workgroups_per_pose = src.size(1);
  auto dst_t = TPack<int32_t, 2, D>::empty({n_poses, workgroups_per_pose});
  auto dst = dst_t.view;
  DO<D>::template foreach_pose_workgroup<launch_t>(
      mgr, n_poses, workgroups_per_pose, [=] EIGEN_DEVICE_FUNC(int wg) {
        int const pose = wg / workgroups_per_pose;
        int const pose_wg = wg % workgroups_per_pose;
        dst[pose][pose_wg] = src[pose][pose_wg] + wg;
      });
  return dst_t;
}

template <tmol::Device D>
auto DevOpsTests<D>::test_scan_inclusive(TView<int32_t, 1, D> src)
    -> TPack<int32_t, 1, D> {
  ContextManager mgr;
  int n = src.size(0);
  auto dst_t = TPack<int32_t, 1, D>::empty({n});
  auto dst = dst_t.view;
  DO<D>::template scan<mgpu::scan_type_inc>(
      mgr, src.data(), dst.data(), n, mgpu::plus_t<int32_t>());
  return dst_t;
}

template <tmol::Device D>
auto DevOpsTests<D>::test_scan_exclusive(TView<int32_t, 1, D> src)
    -> TPack<int32_t, 1, D> {
  ContextManager mgr;
  int n = src.size(0);
  auto dst_t = TPack<int32_t, 1, D>::zeros({n});  // dst[0] = 0 (identity)
  auto dst = dst_t.view;
  DO<D>::template scan<mgpu::scan_type_exc>(
      mgr, src.data(), dst.data(), n, mgpu::plus_t<int32_t>());
  return dst_t;
}

template <tmol::Device D>
auto DevOpsTests<D>::test_scan_and_return_total_inclusive(
    TView<int32_t, 1, D> src) -> std::tuple<TPack<int32_t, 1, D>, int32_t> {
  ContextManager mgr;
  int n = src.size(0);
  auto dst_t = TPack<int32_t, 1, D>::empty({n});
  auto dst = dst_t.view;
  int32_t total = DO<D>::template scan_and_return_total<mgpu::scan_type_inc>(
      mgr, src.data(), dst.data(), n, mgpu::plus_t<int32_t>());
  return {dst_t, total};
}

template <tmol::Device D>
auto DevOpsTests<D>::test_scan_and_return_total_exclusive(
    TView<int32_t, 1, D> src) -> std::tuple<TPack<int32_t, 1, D>, int32_t> {
  ContextManager mgr;
  int n = src.size(0);
  auto dst_t = TPack<int32_t, 1, D>::zeros({n});  // dst[0] = 0 (identity)
  auto dst = dst_t.view;
  int32_t total = DO<D>::template scan_and_return_total<mgpu::scan_type_exc>(
      mgr, src.data(), dst.data(), n, mgpu::plus_t<int32_t>());
  return {dst_t, total};
}

template <tmol::Device D>
auto DevOpsTests<D>::test_reduce(TView<int32_t, 1, D> src) -> int32_t {
  ContextManager mgr;
  return DO<D>::template reduce<int32_t>(
      mgr, src.data(), src.size(0), mgpu::plus_t<int32_t>());
}

template <tmol::Device D>
auto DevOpsTests<D>::test_load_balancing_search(
    TView<int32_t, 1, D> exc_scan_offsets, int n_work_units_total)
    -> TPack<int32_t, 1, D> {
  ContextManager mgr;
  return DO<D>::template load_balancing_search<launch_t>(
      mgr,
      n_work_units_total,
      exc_scan_offsets.data(),
      exc_scan_offsets.size(0));
}

template <tmol::Device D>
auto DevOpsTests<D>::test_segmented_scan_inclusive(
    TView<int32_t, 1, D> src, TView<int32_t, 1, D> seg_starts)
    -> TPack<int32_t, 1, D> {
  ContextManager mgr;
  return DO<D>::template segmented_scan<mgpu::scan_type_inc>(
      mgr,
      src.data(),
      seg_starts.data(),
      src.size(0),
      seg_starts.size(0),
      mgpu::plus_t<int32_t>(),
      0);
}

template <tmol::Device D>
auto DevOpsTests<D>::test_segmented_scan_exclusive(
    TView<int32_t, 1, D> src, TView<int32_t, 1, D> seg_starts)
    -> TPack<int32_t, 1, D> {
  ContextManager mgr;
  return DO<D>::template segmented_scan<mgpu::scan_type_exc>(
      mgr,
      src.data(),
      seg_starts.data(),
      src.size(0),
      seg_starts.size(0),
      mgpu::plus_t<int32_t>(),
      0);
}

}  // namespace device_operations
}  // namespace common
}  // namespace score
}  // namespace tests
}  // namespace tmol
