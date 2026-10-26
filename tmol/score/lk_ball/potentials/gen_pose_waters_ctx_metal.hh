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

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
