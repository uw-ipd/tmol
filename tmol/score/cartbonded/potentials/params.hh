#pragma once

#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "params_types.hh"

namespace tmol {

template <typename Int>
struct enable_tensor_view<
    score::cartbonded::potentials::CartBondedLengthParams<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

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
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

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
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

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
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

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
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

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
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

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
