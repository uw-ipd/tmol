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
// The view rebuilding itself is shared; only the scalar unpacking is
// per-term, because each kernel family names its own launch constants.
template <typename Ctx>
Ctx tmol_build_metal_ctx(device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* scalars = tmol::score::common::mps_scalars<Ctx>(meta);
  ctx.max_n_blocks = int(scalars[Ctx::scalar_max_n_blocks]);
  ctx.output_block_pair_energies =
      scalars[Ctx::scalar_output_block_pair_energies] != 0;
  ctx.accumulate_derivs = scalars[Ctx::scalar_accumulate_derivs] != 0;
  ctx.max_n_upper_triangle_inds =
      int(scalars[Ctx::scalar_max_n_upper_triangle_inds]);
  ctx.use_shared_compact_block_neighbors =
      scalars[Ctx::scalar_use_shared_compact_block_neighbors] != 0;
  return ctx;
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
