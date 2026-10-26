#pragma once

// Metal-side spelling of the sphere-overlap kernel contexts.
//
// sphere_overlap_ctx.hh declares the field lists once, templated on the
// view; this binds them to tmol::portable::MetalView and is the only one a
// `.metal` translation unit includes, because TView reaches the torch tensor
// layer.

#include <tmol/score/common/mps_ctx_builder.hh>
#include <tmol/score/common/mps_float_scalar.hh>
#include <tmol/score/common/sphere_overlap_ctx.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace common {
namespace sphere_overlap {

template <typename Real, typename Int>
using BlockSpheresCtxMetal =
    BlockSpheresCtxT<tmol::portable::MetalView, Real, Int>;

template <typename Real, typename Int>
using BlockNeighborsCtxMetal =
    BlockNeighborsCtxT<tmol::portable::MetalView, Real, Int>;

template <typename Real, typename Int>
using RotSpheresCtxMetal = RotSpheresCtxT<tmol::portable::MetalView, Real, Int>;

template <typename Real, typename Int>
using BlockSpheresFromRotCtxMetal =
    BlockSpheresFromRotCtxT<tmol::portable::MetalView, Real, Int>;

template <typename Real, typename Int>
using RotPairListCtxMetal =
    RotPairListCtxT<tmol::portable::MetalView, Real, Int>;

}  // namespace sphere_overlap
}  // namespace common
}  // namespace score
}  // namespace tmol
