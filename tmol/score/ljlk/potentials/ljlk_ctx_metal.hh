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

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
