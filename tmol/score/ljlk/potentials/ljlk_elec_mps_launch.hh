#pragma once

// The one call the fused LJ/LK + elec dispatch makes into the Metal backend.
//
// ljlk_elec_pose_score.impl.hh is compiled for every backend, so it cannot
// name Metal-only types. This header gives it a device-templated entry that
// compiles to nothing unless D is MPS -- and to nothing at all in a build
// without the Metal backend.

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/score/common/mps_launch.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/ljlk/potentials/ljlk_elec_mps_kernels_source.h>
#endif

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

// One threadgroup per rotamer pair in the context's dispatch list.
template <tmol::Device D, typename Ctx>
void ljlk_elec_mps_launch_if(Ctx& ctx, int64_t n_workgroups) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    namespace sc = tmol::score::common;
    auto args = sc::mps_collect(ctx);
    int64_t const group = sc::TMOL_MPS_THREADGROUP_SIZE;
    sc::mps_launch(
        kLJLKElecMetalShaderSource,
        "ljlk_elec_eval_weighted_rotamer_energies_f32",
        args,
        n_workgroups * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached the fused LJ/LK-elec term, but this "
        "build has no Metal backend");
#endif
  } else {
    (void)ctx;
    (void)n_workgroups;
  }
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
