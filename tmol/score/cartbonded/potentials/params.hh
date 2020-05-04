#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

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

namespace tmol {

template <typename Int>
struct enable_tensor_view<
    score::cartbonded::potentials::CartBondedLengthParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Int>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::cartbonded::potentials::CartBondedLengthParams<Int>)
                     / sizeof(Int)
               : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Int>
struct enable_tensor_view<
    score::cartbonded::potentials::CartBondedAngleParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Int>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::cartbonded::potentials::CartBondedAngleParams<Int>)
                     / sizeof(Int)
               : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Int>
struct enable_tensor_view<
    score::cartbonded::potentials::CartBondedTorsionParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Int>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(score::cartbonded::potentials::CartBondedTorsionParams<
                        Int>)
                     / sizeof(Int)
               : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::cartbonded::potentials::CartBondedHarmonicTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::cartbonded::potentials::
                                 CartBondedHarmonicTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::cartbonded::potentials::CartBondedPeriodicTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::cartbonded::potentials::
                                 CartBondedPeriodicTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::cartbonded::potentials::CartBondedSinusoidalTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::cartbonded::potentials::
                                 CartBondedSinusoidalTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
