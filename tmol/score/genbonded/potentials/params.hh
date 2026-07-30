#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

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
namespace tmol {

template <typename Int>
struct enable_tensor_view<
    score::genbonded::potentials::GenBondedTorsionParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }
  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::genbonded::potentials::GenBondedTorsionParams<Int>)
                     / sizeof(Int)
               : 0;
  }
  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::genbonded::potentials::GenBondedTorsionTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }
  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::genbonded::potentials::GenBondedTorsionTypeParams<
                         Real>)
                     / sizeof(Real)
               : 0;
  }
  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
