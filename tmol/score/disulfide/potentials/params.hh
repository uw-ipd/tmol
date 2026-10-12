#include <tmol/utility/tensor/native.hh>
#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include "params_core.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template <typename T, int N>
using Vec = Eigen::Matrix<T, N, 1>;

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Real>
struct enable_tensor_view<
    score::disulfide::potentials::DisulfideGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0)
               ? sizeof(
                     score::disulfide::potentials::DisulfideGlobalParams<Real>)
                     / sizeof(Real)
               : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
