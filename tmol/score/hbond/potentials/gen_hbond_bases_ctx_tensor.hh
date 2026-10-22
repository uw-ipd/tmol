#pragma once

// Host-side spelling of the derived-atom pre-pass context. A Metal
// translation unit binds the same template to tmol::portable::MetalView
// instead and never includes this file.

#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_ctx.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, typename Int, tmol::Device D>
using HBondBasesCtx = HBondBasesCtxT<
    tmol::score::common::TViewBinder<D>::template View,
    Real,
    Int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
