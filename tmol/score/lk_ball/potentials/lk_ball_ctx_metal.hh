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
using LKBallCtxMetal =
    LKBallCtxT<tmol::portable::MetalView, Real, Int, OutRank>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
