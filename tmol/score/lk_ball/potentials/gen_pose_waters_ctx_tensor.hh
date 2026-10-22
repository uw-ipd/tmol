#pragma once

// Host-side spelling of the water-generation kernel context.
//
// gen_pose_waters_ctx.hh declares the field list once, templated on the view.
// This binds it to tmol::TView for the CPU and CUDA builds; a Metal
// translation unit binds the same template to tmol::portable::MetalView and
// never includes this file.

#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_ctx.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, typename Int, tmol::Device D>
using WaterGenCtx =
    WaterGenCtxT<tmol::score::common::TViewBinder<D>::template View, Real, Int>;

template <typename Real, typename Int, tmol::Device D>
using WaterGenDerivCtx = WaterGenDerivCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
