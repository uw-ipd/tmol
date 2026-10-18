#pragma once

// The one call the generic water-generation dispatch makes into the Metal
// backend. See lk_ball_mps_launch.hh for the same shape on the scoring side.

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/score/common/mps_launch.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_mps_kernels_source.h>
#endif

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <tmol::Device Dev, typename Ctx>
void water_gen_mps_launch_if(Ctx& ctx, int64_t n_workgroups) {
  if constexpr (Dev == tmol::Device::MPS) {
#ifdef WITH_MPS
    namespace sc = tmol::score::common;
    auto args = sc::mps_collect(ctx);
    args.set_scalar(Ctx::n_views, Ctx::n_rots, ctx.n_rots_total);

    int64_t const group = sc::TMOL_MPS_THREADGROUP_SIZE;
    sc::mps_launch(
        kWaterGenMetalShaderSource,
        "water_gen_build_waters_f32",
        args,
        n_workgroups * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached LK-ball water generation, but this "
        "build has no Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_workgroups;
  }
}

// The backward pass. Its context carries the three torsion-table lengths as
// scalars, because the tables themselves are packed into one view.
template <tmol::Device Dev, typename Ctx>
void water_gen_derivs_mps_launch_if(Ctx& ctx, int64_t n_workgroups) {
  if constexpr (Dev == tmol::Device::MPS) {
#ifdef WITH_MPS
    namespace sc = tmol::score::common;
    auto args = sc::mps_collect(ctx);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_n_sp2_water_tors, ctx.n_sp2_water_tors);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_n_sp3_water_tors, ctx.n_sp3_water_tors);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_n_ring_water_tors, ctx.n_ring_water_tors);

    int64_t const group = sc::TMOL_MPS_THREADGROUP_SIZE;
    sc::mps_launch(
        kWaterGenMetalShaderSource,
        "water_gen_build_water_derivs_f32",
        args,
        n_workgroups * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached LK-ball water generation, but this "
        "build has no Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_workgroups;
  }
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
