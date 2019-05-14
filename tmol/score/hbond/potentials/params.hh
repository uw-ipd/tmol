#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real>
struct HBondPoly {
  Eigen::Matrix<Real, 11, 1> coeffs;
  Real zero_pad;  // Eigen wants to align to memory boundaries
  Eigen::Matrix<Real, 2, 1> range;
  Eigen::Matrix<Real, 2, 1> bound;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Real>
struct enable_tensor_view<score::hbond::potentials::HBondPoly<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::hbond::potentials::HBondPoly<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
