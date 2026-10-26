#pragma once

// The Metal instantiation of the annealer context.
//
// A Metal translation unit cannot include the torch tensor layer, so it binds
// the same field list to MetalView instead of TView. See annealer_ctx.hh.

#include <tmol/pack/compiled/annealer_ctx.hh>
#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/common/mps_float_scalar.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <typename Real, typename Int>
using AnnealerCtxMetal = AnnealerCtxT<tmol::portable::MetalView, Real, Int>;

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
