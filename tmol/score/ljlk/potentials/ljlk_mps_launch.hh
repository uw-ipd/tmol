#pragma once

// The one call the generic LJ/LK dispatch makes into the Metal backend.
//
// ljlk_pose_score.impl.hh is compiled for every backend, so it cannot name
// Metal-only types. This header gives it a single device-templated entry
// that compiles to nothing unless D is MPS -- and to nothing at all in a
// build without the Metal backend.

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/score/common/mps_float_scalar.hh>
#include <tmol/score/common/mps_launch.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/ljlk/potentials/ljlk_mps_kernels_source.h>
#endif

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

// Launch a named Metal entry point over a context.
//
// n_workgroups threadgroups; the kernel body reads its workgroup index from
// the entry's threadgroup_position_in_grid, which is the same index
// foreach_pose_workgroup hands the CPU and CUDA lambdas.
template <tmol::Device D, typename Ctx>
void ljlk_mps_launch_if(
    char const* kernel_name, Ctx& ctx, int64_t n_workgroups) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    namespace sc = tmol::score::common;
    auto args = sc::mps_collect(ctx);
    args.set_scalar(Ctx::n_views, Ctx::scalar_max_n_blocks, ctx.max_n_blocks);
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_max_n_upper_triangle_inds,
        ctx.max_n_upper_triangle_inds);
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_output_block_pair_energies,
        ctx.output_block_pair_energies ? 1 : 0);
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_use_shared_compact_block_neighbors,
        ctx.use_shared_compact_block_neighbors ? 1 : 0);
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_require_gradient,
        ctx.require_gradient ? 1 : 0);
    // max_dis is the one non-integral launch constant; the metadata buffer is
    // int64, so it travels as the float's bit pattern. See mps_float_scalar.hh.
    args.set_scalar(
        Ctx::n_views,
        Ctx::scalar_max_dis,
        sc::tmol_pack_float(float(ctx.max_dis)));

    int64_t const group = sc::TMOL_MPS_THREADGROUP_SIZE;
    sc::mps_launch(
        kLJLKMetalShaderSource,
        kernel_name,
        args,
        n_workgroups * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached the LJ/LK term, but this build has no "
        "Metal backend");
#endif
  } else {
    (void)kernel_name;
    (void)ctx;
    (void)n_workgroups;
  }
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
