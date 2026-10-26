#pragma once

// Metal-side spelling of the derived-atom pre-pass context.

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases_ctx.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, typename Int>
using HBondBasesCtxMetal = HBondBasesCtxT<tmol::portable::MetalView, Real, Int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
