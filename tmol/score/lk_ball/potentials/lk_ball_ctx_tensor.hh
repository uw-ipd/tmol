#pragma once

// Host-side spelling of the LK-ball kernel context.
//
// lk_ball_ctx.hh declares the field list once, templated on the view. This
// header binds it to tmol::TView for the CPU and CUDA builds; a Metal
// translation unit binds the same template to tmol::portable::MetalView and
// never includes this file.

#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_ctx.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, typename Int, tmol::Device D, int OutRank = 4>
using LKBallCtx = LKBallCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int,
    OutRank>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
