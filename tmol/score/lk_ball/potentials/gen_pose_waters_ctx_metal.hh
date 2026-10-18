#pragma once

// Metal-side spelling of the water-generation kernel context.

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_ctx.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, typename Int>
using WaterGenCtxMetal = WaterGenCtxT<tmol::portable::MetalView, Real, Int>;

template <typename Real, typename Int>
using WaterGenDerivCtxMetal =
    WaterGenDerivCtxT<tmol::portable::MetalView, Real, Int>;

template <typename Ctx>
Ctx water_gen_build_metal_deriv_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* scalars = tmol::score::common::mps_scalars<Ctx>(meta);
  ctx.n_sp2_water_tors = int(scalars[Ctx::scalar_n_sp2_water_tors]);
  ctx.n_sp3_water_tors = int(scalars[Ctx::scalar_n_sp3_water_tors]);
  ctx.n_ring_water_tors = int(scalars[Ctx::scalar_n_ring_water_tors]);
  return ctx;
}

template <typename Ctx>
Ctx water_gen_build_metal_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* scalars = tmol::score::common::mps_scalars<Ctx>(meta);
  ctx.n_rots_total = int(scalars[Ctx::n_rots]);
  return ctx;
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
