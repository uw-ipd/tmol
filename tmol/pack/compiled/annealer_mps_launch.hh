#pragma once

// The call the generic annealer dispatch makes into the Metal backend.
//
// annealer.hh is compiled for every backend, so it cannot name Metal-only
// types. This entry compiles to nothing unless D is MPS -- and to nothing at
// all in a build without the Metal backend.

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/pack/compiled/annealer_ctx.hh>
#include <tmol/pack/compiled/annealer_mps_kernels_source.h>
#include <tmol/score/common/mps_launch.hh>
#include <tmol/score/common/workgroup_lane.hh>
#endif

namespace tmol {
namespace pack {
namespace compiled {

// One workgroup per trajectory; n_units is n_poses * n_traj.
template <tmol::Device D, typename Ctx>
void annealer_mps_launch_if(Ctx& ctx, int64_t n_units) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    namespace sc = tmol::score::common;
    auto args = sc::mps_collect(ctx);
    // The launch scalars ride in the metadata buffer, in the order
    // build_annealer_metal_ctx unpacks them.
    args.set_scalar(Ctx::n_views, Ctx::scalar_chunk_size, ctx.chunk_size);
    args.set_scalar(Ctx::n_views, Ctx::scalar_max_n_res, ctx.max_n_res);
    args.set_scalar(Ctx::n_views, Ctx::scalar_n_traj, ctx.n_traj);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_n_outer_iterations, ctx.n_outer_iterations);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_n_inner_iterations, ctx.n_inner_iterations);
    args.set_scalar(Ctx::n_views, Ctx::scalar_seed_lo, ctx.seed_lo);
    args.set_scalar(Ctx::n_views, Ctx::scalar_seed_hi, ctx.seed_hi);
    args.set_scalar(Ctx::n_views, Ctx::scalar_phase, ctx.phase);
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_high_temp, sc::tmol_pack_float(ctx.high_temp));
    args.set_scalar(
        Ctx::n_views, Ctx::scalar_low_temp, sc::tmol_pack_float(ctx.low_temp));
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_cooling_factor,
        sc::tmol_pack_float(ctx.cooling_factor));

    int64_t const group = sc::TMOL_MPS_THREADGROUP_SIZE;
    sc::mps_launch(
        kAnnealerMetalShaderSource,
        "annealer_run_f32",
        args,
        n_units * group,
        group);
#else
    throw std::runtime_error(
        "tmol: simulated annealing reached the Metal path, but this build "
        "has no Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_units;
  }
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
