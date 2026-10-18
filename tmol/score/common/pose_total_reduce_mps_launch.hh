#pragma once

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/score/common/mps_launch.hh>
#include <tmol/score/common/pose_total_reduce_mps_kernels_source.h>
#include <tmol/score/common/workgroup_lane.hh>
#endif

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D, typename Ctx>
void pose_total_reduce_mps_launch_if(Ctx& ctx, int64_t n_workgroups) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    auto args = mps_collect(ctx);
    args.set_scalar(Ctx::n_views, Ctx::scalar_n_cells, ctx.n_cells);
    int64_t const group = TMOL_MPS_THREADGROUP_SIZE;
    mps_launch(
        kPoseTotalReduceMetalShaderSource,
        "pose_total_reduce_f32",
        args,
        n_workgroups * group,
        group);
#else
    throw std::runtime_error(
        "tmol: pose-total reduction reached the Metal path, but this build "
        "has no Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_workgroups;
  }
}

}  // namespace common
}  // namespace score
}  // namespace tmol
