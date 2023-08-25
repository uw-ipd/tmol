#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace score {
namespace backbone_torsion {
namespace potentials {

template <typename T, int N>
using Vec = Eigen::Matrix<T, N, 1>;

template <typename Int>
struct RamaParameters {
  Int phis[4];
  Int psis[4];
  Int table_index;
};

template <typename Real>
struct RamaTableParams {
  Real bbstarts[2];
  Real bbsteps[2];
};

template <typename Real>
struct OmegaParameters {
  Real atoms[4];
  Real K;
};

template <typename Real>
struct OmegaGlobalParams {
  Real K;
};

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Int>
struct enable_tensor_view<
    score::backbone_torsion::potentials::RamaParameters<Int>> {
  static const bool enabled = enable_tensor_view<Int>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Int>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::backbone_torsion::potentials::RamaParameters<Int>)
                     / sizeof(Int)
               : 0;
  }

  typedef typename enable_tensor_view<Int>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::backbone_torsion::potentials::RamaTableParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::backbone_torsion::potentials::RamaTableParams<Real>)
                     / sizeof(Real)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::backbone_torsion::potentials::OmegaParameters<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::backbone_torsion::potentials::OmegaParameters<Real>)
                     / sizeof(Real)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<
    score::backbone_torsion::potentials::OmegaGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(score::backbone_torsion::potentials::OmegaGlobalParams<
                        Real>)
                     / sizeof(Real)
               : 0;
  }
};

}  // namespace tmol
