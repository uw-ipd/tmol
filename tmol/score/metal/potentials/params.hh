#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace score {
namespace metal {
namespace potentials {

// one row per (metal block type, donor key)
template <typename Real>
struct MetalSiteParams {
  Real d0;
  Real well_depth;
  Real radial_sd;
  Real lateral_sd;
};

// one row per (metal block type, fan atom pair)
template <typename Real>
struct MetalFanParams {
  Real l0;
  Real sd;
};

}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Real>
struct enable_tensor_view<score::metal::potentials::MetalSiteParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::metal::potentials::MetalSiteParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::metal::potentials::MetalFanParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::metal::potentials::MetalFanParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
