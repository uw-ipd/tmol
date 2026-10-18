#pragma once

// Host-side spelling of the elec kernel context.
//
// elec_ctx.hh declares the field list once, templated on the view. This header
// binds it to tmol::TView for the CPU and CUDA builds; a Metal translation
// unit binds the same template to tmol::portable::MetalView instead and never
// includes this file.

#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/elec/potentials/elec_ctx.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real, typename Int, tmol::Device D, int OutRank = 4>
using ElecCtx = ElecCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int,
    OutRank>;

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
