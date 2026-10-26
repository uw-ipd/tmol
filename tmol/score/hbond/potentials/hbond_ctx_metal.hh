#pragma once

// Metal-side spelling of the hbond kernel context.
//
// hbond_ctx.hh declares the field list once, templated on the view.
// hbond_ctx_tensor.hh binds it to tmol::TView for the CPU and CUDA builds;
// this header binds the same template to tmol::portable::MetalView and is the
// only one a `.metal` translation unit includes, because TView reaches the
// torch tensor layer.

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/hbond/potentials/hbond_ctx.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <
    typename Real,
    typename Int,
    int OutRank,
    int DispRank,
    int DerivRank = 2>
using HBondCtxMetal = HBondCtxT<
    tmol::portable::MetalView,
    Real,
    Int,
    typename hbond_poly_scalar<tmol::Device::MPS>::type,
    OutRank,
    DispRank,
    DerivRank>;

// Build a context from the bound buffers and the metadata.
//
}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
