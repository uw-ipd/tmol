#pragma once

// Host-side spelling of the LJ/LK kernel context.
//
// ljlk_ctx.hh declares the field list once, templated on the view. This
// header binds it to tmol::TView for the CPU and CUDA builds; a Metal
// translation unit binds the same template to tmol::portable::MetalView
// instead and never includes this file.

#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/ljlk/potentials/ljlk_ctx.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, typename Int, tmol::Device D, int OutRank = 4>
using LJLKCtx = LJLKCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int,
    OutRank>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
