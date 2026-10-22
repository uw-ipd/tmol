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
namespace genbonded {
namespace potentials {

// Indices of the four atoms defining a proper torsion within a block.
template <typename Int>
struct GenBondedTorsionParams {
  Int atom_index_i;
  Int atom_index_j;
  Int atom_index_k;
  Int atom_index_l;
  // Row index into the genbonded_torsion_params tensor that holds the
  // actual force-field coefficients for this torsion.
  Int param_index;
};

// Force-field coefficients for one torsion type.
// Energy function (to be defined in potentials.hh):
//   E = k1*(1 + cos(theta - offset))
//     + k2*(1 + cos(2*theta - offset))
//     + k3*(1 + cos(3*theta - offset))
//     + k4*(1 + cos(4*theta - offset))
//
// TODO: confirm sign conventions and offset semantics with the training code.
template <typename Real>
struct GenBondedTorsionTypeParams {
  Real k1;
  Real k2;
  Real k3;
  Real k4;
  Real offset;
};

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol

// ---------------------------------------------------------------------------
// TensorUtil specialisations so the structs can be used as tensor elements.
// ---------------------------------------------------------------------------
