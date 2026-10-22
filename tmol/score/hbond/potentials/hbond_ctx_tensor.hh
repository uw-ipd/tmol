#pragma once

// Host-side spelling of the hbond kernel context.
//
// hbond_ctx.hh declares the field list once, templated on the view. This
// header binds it to tmol::TView for the CPU and CUDA builds; a Metal
// translation unit binds the same template to tmol::portable::MetalView
// instead and never includes this file.

#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/hbond/potentials/hbond_ctx.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <
    typename Real,
    typename Int,
    tmol::Device D,
    int OutRank = 4,
    int DispRank = 3,
    int DerivRank = 2>
using HBondCtx = HBondCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int,
    typename hbond_poly_scalar<D>::type,
    OutRank,
    DispRank,
    DerivRank>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
