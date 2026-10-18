#pragma once

// Metal-side spelling of the LJ/LK kernel context.
//
// ljlk_ctx.hh declares the field list once, templated on the view;
// ljlk_ctx_tensor.hh binds it to tmol::TView for the CPU and CUDA builds.
// This binds the same template to tmol::portable::MetalView and is the only
// one a `.metal` translation unit includes, because TView reaches the torch
// tensor layer.

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/common/mps_float_scalar.hh>
#include <tmol/score/ljlk/potentials/ljlk_ctx.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, typename Int, int OutRank>
using LJLKCtxMetal = LJLKCtxT<tmol::portable::MetalView, Real, Int, OutRank>;

template <typename Ctx>
Ctx ljlk_build_metal_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* scalars =
      tmol::score::common::mps_scalars<Ctx>(meta);
  ctx.max_n_blocks = int(scalars[Ctx::scalar_max_n_blocks]);
  ctx.max_n_upper_triangle_inds =
      int(scalars[Ctx::scalar_max_n_upper_triangle_inds]);
  ctx.output_block_pair_energies =
      scalars[Ctx::scalar_output_block_pair_energies] != 0;
  ctx.use_shared_compact_block_neighbors =
      scalars[Ctx::scalar_use_shared_compact_block_neighbors] != 0;
  ctx.require_gradient = scalars[Ctx::scalar_require_gradient] != 0;
  ctx.max_dis = tmol::score::common::tmol_unpack_float(
      scalars[Ctx::scalar_max_dis]);
  return ctx;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
