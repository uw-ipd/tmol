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
namespace cartbonded {
namespace potentials {

template <typename Int>
struct CartBondedLengthParams {
  Int atom_index_i;
  Int atom_index_j;
  Int param_index;
};

template <typename Int>
struct CartBondedAngleParams {
  Int atom_index_i;
  Int atom_index_j;
  Int atom_index_k;
  Int param_index;
};

template <typename Int>
struct CartBondedTorsionParams {
  Int atom_index_i;
  Int atom_index_j;
  Int atom_index_k;
  Int atom_index_l;
  Int param_index;
};

template <typename Real>
struct CartBondedHarmonicTypeParams {
  Real K;
  Real x0;
};

template <typename Real>
struct CartBondedPeriodicTypeParams {
  Real K;
  Real x0;
  Real period;
};

template <typename Real>
struct CartBondedSinusoidalTypeParams {
  Real k1;
  Real k2;
  Real k3;
  Real phi1;
  Real phi2;
  Real phi3;
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
