#pragma once

// Reduce per-block-pair energy cells to one total per pose.
//
// The GPU backends score a block pair per workgroup and write the result to
// its own cell rather than atomically adding into a shared total, so that the
// pose total does not depend on the order the workgroups happen to finish in.
// This is the second half of that: a reduction whose traversal order is fixed
// and whose arithmetic is compensated, so the float32 total it writes is very
// nearly the correctly-rounded sum of the cells.
//
// Both properties are needed. A fixed order alone gives reproducibility but
// still lets the total jump by several ULP when a coordinate moves by far
// less than one, because a plain float32 accumulation's error term moves with
// its inputs; a line search that finite-differences the energy reads those
// jumps as slope. Compensating the sum removes them.
//
// One workgroup per (term, pose). Each lane walks a fixed stride of the cells,
// then lane 0 merges the per-lane partials in lane order.

#include <tmol/score/common/pose_total_reduce_ctx.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/utility/portable/compensated.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace common {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    int N_T,
    typename Ctx,
    typename Shared>
TMOL_HD void pose_total_reduce_body(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<D> lane = {}) {
  typedef tmol::portable::CompensatedSum<Real> Sum;

  int const n_poses = ctx.cells.size(1);
  int const n_blocks = ctx.cells.size(2);
  int const term = cta / n_poses;
  int const pose = cta - term * n_poses;

  auto accumulate_stride = ([&](int tid) {
    Sum s;
    for (int i = tid; i < ctx.n_cells; i += N_T) {
      s.add(ctx.cells[term][pose][i / n_blocks][i - (i / n_blocks) * n_blocks]);
    }
    shared.total[tid] = s.total;
    shared.correction[tid] = s.correction;
  });
  DeviceDispatch<D>::template for_each_in_workgroup<N_T>(
      accumulate_stride, lane);
  DeviceDispatch<D>::synchronize_workgroup(lane);

  auto merge = ([&](int tid) {
    if (tid != 0) return;
    Sum s;
    for (int i = 0; i < N_T; ++i) {
      Sum part;
      part.total = shared.total[i];
      part.correction = shared.correction[i];
      s.merge(part);
    }
    ctx.totals[term][pose][0][0] = s.value();
  });
  DeviceDispatch<D>::template for_each_in_workgroup<N_T>(merge, lane);
}

template <typename Real, int N_T>
struct PoseTotalReduceShared {
#if !TMOL_PORTABLE_METAL
  PoseTotalReduceShared() {}
#endif
  Real total[N_T];
  Real correction[N_T];
};

}  // namespace common
}  // namespace score
}  // namespace tmol
