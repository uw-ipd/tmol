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

// This context has no scalar slots, so building it is just the shared view
// rebuilding.
template <typename Ctx>
Ctx tmol_build_bases_metal_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  return tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
