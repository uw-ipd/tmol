#pragma once

// The one call the generic derived-atom pre-pass makes into the Metal backend.
//
// gen_hbond_bases.impl.hh is compiled for every backend, so it cannot name
// Metal-only types. This header gives it a single device-templated entry that
// compiles to nothing unless D is MPS -- and to nothing at all in a build
// without the Metal backend.

#include <tmol/utility/portable/device.hh>

#include <cstdint>
#include <stdexcept>

#ifdef WITH_MPS
#include <tmol/score/common/mps_launch.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_mps_kernels_source.h>
#endif

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <tmol::Device D, typename Ctx>
void hbond_bases_mps_launch_if(
    char const* kernel_name, Ctx& ctx, int64_t n_workgroups) {
  if constexpr (D == tmol::Device::MPS) {
#ifdef WITH_MPS
    namespace sc = tmol::score::common;
    auto args = sc::mps_collect(ctx);
    int64_t const group = sc::TMOL_MPS_THREADGROUP_SIZE;
    sc::mps_launch(
        kHBondBasesMetalShaderSource,
        kernel_name,
        args,
        n_workgroups * group,
        group);
#else
    throw std::runtime_error(
        "tmol: an MPS tensor reached the hbond derived-atom pre-pass, but "
        "this build has no Metal backend");
#endif
  } else {
    (void)kernel_name;
    (void)ctx;
    (void)n_workgroups;
  }
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
