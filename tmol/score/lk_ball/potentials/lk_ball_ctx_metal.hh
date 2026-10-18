#pragma once

// Metal-side spelling of the LK-ball kernel context.
//
// lk_ball_ctx.hh declares the field list once, templated on the view;
// lk_ball_ctx_tensor.hh binds it to tmol::TView for the CPU and CUDA builds.
// This binds the same template to tmol::portable::MetalView and is the only
// one a `.metal` translation unit includes, because TView reaches the torch
// tensor layer.

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_ctx.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, typename Int, int OutRank>
using LKBallCtxMetal = LKBallCtxT<tmol::portable::MetalView, Real, Int, OutRank>;

template <typename Ctx>
Ctx lk_ball_build_metal_ctx(
    device void* const* ptrs, device const int64_t* meta) {
  Ctx ctx = tmol::score::common::mps_build_ctx<Ctx>(ptrs, meta);
  device const int64_t* scalars = tmol::score::common::mps_scalars<Ctx>(meta);
  ctx.max_n_blocks = int(scalars[Ctx::scalar_max_n_blocks]);
  ctx.n_block_pairs = int(scalars[Ctx::scalar_n_block_pairs]);
  ctx.output_block_pair_energies =
      scalars[Ctx::scalar_output_block_pair_energies] != 0;
  ctx.use_shared_compact_block_neighbors =
      scalars[Ctx::scalar_use_shared_compact_block_neighbors] != 0;
  // Read by the derivative entry only: it decides whether dTdV is indexed
  // per block pair or per pose, and reading it as garbage walks off the end
  // of a per-pose dTdV.
  ctx.block_pair_scoring = scalars[Ctx::scalar_block_pair_scoring] != 0;
  return ctx;
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
