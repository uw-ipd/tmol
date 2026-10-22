#pragma once

// Host-side spelling of the annealer context.
//
// annealer_ctx.hh declares the field list once, templated on the view. This
// header binds it to tmol::TView for the CPU and CUDA builds; a Metal
// translation unit binds the same template to tmol::portable::MetalView and
// never includes this file.

#include <tmol/pack/compiled/annealer_ctx.hh>
#include <tmol/score/common/tview_binder.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <typename Real, typename Int, tmol::Device D>
using AnnealerCtx =
    AnnealerCtxT<tmol::score::common::TViewBinder<D>::template View, Real, Int>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
