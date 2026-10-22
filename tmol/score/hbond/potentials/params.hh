#pragma once

#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/tensor/native.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "params_types.hh"

namespace tmol {

// A double-float occupies two primitive slots. Declared here rather than
// beside the type because hbond is its only tensor-resident user; move it to
// the tensor layer if a second term needs it.
template <typename Real>
struct enable_tensor_view<tmol::portable::DoubleFloat<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) { return (i == 0) ? 2 : 0; }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::hbond::potentials::HBondPoly<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    enable_tensor_view<Real>::scalar_type;
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    // Primitive units, not Real units: the two agree when Real is a scalar
    // and differ when it is a double-float, which is two float32 slots.
    return (i == 0)
               ? sizeof(score::hbond::potentials::HBondPoly<Real>)
                     / sizeof(typename enable_tensor_view<Real>::PrimitiveType)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::hbond::potentials::HBondPolynomials<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(score::hbond::potentials::HBondPolynomials<Real>)
                     / sizeof(typename enable_tensor_view<Real>::PrimitiveType)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::hbond::potentials::HBondGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::hbond::potentials::HBondGlobalParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::hbond::potentials::HBondPairParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::hbond::potentials::HBondPairParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
