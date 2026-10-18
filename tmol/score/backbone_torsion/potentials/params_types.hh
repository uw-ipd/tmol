#pragma once

// Plain parameter structs, free of the torch tensor layer.
//
// Split out of params.hh so that device-side scalar math can include the
// struct definitions without dragging in TensorUtil.h (and through it Eigen
// and torch), which exist only for the enable_tensor_view specializations.
// A Metal kernel receives these as bound buffers and needs the layout, not
// the tensor machinery. See tmol/tests/utility/portable/.

#include <tmol/utility/portable/matrix.hh>

namespace tmol {
namespace score {
namespace backbone_torsion {
namespace potentials {

using tmol::portable::Vec;

template <typename Real>
struct RamaTableParams {
  Real bbstarts[2];
  Real bbsteps[2];
};

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
