#pragma once

// Plain parameter structs, free of the torch tensor layer.
//
// Split out of params.hh so that device-side scalar math can include the
// struct definitions without dragging in TensorUtil.h (and through it Eigen
// and torch), which exist only for the enable_tensor_view specializations.
// A Metal kernel receives these as bound buffers and needs the layout, not
// the tensor machinery. See tmol/tests/utility/portable/.

namespace tmol {
namespace score {
namespace metal {
namespace potentials {

// one row per (metal block type, donor key)
template <typename Real>
struct MetalSiteParams {
  Real d0;
  Real well_depth;
  Real radial_sd;
  Real lateral_sd;
};

// one row per (metal block type, fan atom pair)
template <typename Real>
struct MetalFanParams {
  Real l0;
  Real sd;
};

// one row per donor key: the smallest metal-donor-metal angle a bridge takes,
// and the width of the wall below it, both in radians
template <typename Real>
struct MetalBridgeParams {
  Real floor;
  Real width;
};

}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol
