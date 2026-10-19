#pragma once

// Device-generic entry point for the pose-total reduction.
//
// One call site, three backends. CPU and CUDA reach the body through
// DeviceOperations the way every other tmol kernel does; MPS reaches the same
// body through a named Metal entry, because Metal cannot launch a device
// lambda. The body itself lives in pose_total_reduce_kernels.hh and is
// compiled once per backend from the same source.

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/pose_total_reduce_kernels.hh>
#include <tmol/score/common/pose_total_reduce_mps_launch.hh>
#include <tmol/score/common/pose_total_reduce_ctx_tensor.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace common {

// Sum cells[term][pose][b1][b2] over the trailing two dimensions into
// totals[term][pose][0][0], in a fixed order and with compensated arithmetic.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
struct pose_total_reduce {
  static void f(
      ContextManager& mgr,
      TView<Real, 4, D> cells,
      TView<Real, 4, D> totals) {
    LAUNCH_BOX_32;
    CTA_LAUNCH_T_PARAMS;

    int const n_terms = cells.size(0);
    int const n_poses = cells.size(1);
    int const n_blocks = cells.size(2);

    PoseTotalReduceCtx<Real, D> ctx;
    ctx.cells = cells;
    ctx.totals = totals;
    ctx.n_cells = n_blocks * n_blocks;

    int const n_workgroups = n_terms * n_poses;
    if (D == tmol::Device::MPS) {
      pose_total_reduce_mps_launch_if<D>(ctx, n_workgroups);
      return;
    }

    typedef PoseTotalReduceShared<Real, nt> shared_t;
    auto reduce_one = ([=] TMOL_DEVICE_FUNC(int cta) {
      SHARED_MEMORY shared_t shared;
      pose_total_reduce_body<DeviceDispatch, D, Real, nt>(ctx, shared, cta);
    });
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        mgr, n_workgroups, reduce_one);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
