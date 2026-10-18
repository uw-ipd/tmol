#pragma once

// Metal-side spelling of the elec kernel context.
//
// elec_ctx.hh declares the field list once, templated on the view.
// elec_ctx_tensor.hh binds it to tmol::TView for the CPU and CUDA builds;
// this header binds the same template to tmol::portable::MetalView and is the
// only one a `.metal` translation unit includes, because TView reaches the
// torch tensor layer.
//
// The bound struct doubles as the kernel's argument buffer: every field is
// either a `device` pointer or a scalar, which is exactly what Metal lets a
// buffer-backed struct hold. The host writes MTLBuffer.gpuAddress values into
// the matching slots -- see elec_mps_dispatch.mm, which derives the slot order
// from the same field list rather than repeating it.

#include <tmol/score/elec/potentials/elec_ctx.hh>
#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real, typename Int, int OutRank>
using ElecCtxMetal =
    ElecCtxT<tmol::portable::MetalView, Real, Int, OutRank>;

// Build a context from the bound buffers and the metadata.
//
// The view rebuilding itself is shared; only the scalar unpacking is
// per-term, because each kernel family names its own launch constants.
template <typename Ctx>
Ctx tmol_build_metal_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* scalars = tmol::score::common::mps_scalars<Ctx>(meta);
  ctx.max_n_blocks = int(scalars[Ctx::scalar_max_n_blocks]);
  ctx.max_n_upper_triangle_inds =
      int(scalars[Ctx::scalar_max_n_upper_triangle_inds]);
  ctx.output_block_pair_energies =
      scalars[Ctx::scalar_output_block_pair_energies] != 0;
  ctx.use_shared_compact_block_neighbors =
      scalars[Ctx::scalar_use_shared_compact_block_neighbors] != 0;
  return ctx;
}

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
