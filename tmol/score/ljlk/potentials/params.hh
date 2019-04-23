#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
struct LJTypeParams {
  Real lj_radius;
  Real lj_wdepth;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
};

template <typename Real>
struct LKTypeParams {
  Real lj_radius;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;
};

template <typename Real>
struct LJGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Real>
struct enable_tensor_view<score::ljlk::potentials::LJTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::ljlk::potentials::LJTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::ljlk::potentials::LKTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::ljlk::potentials::LKTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::ljlk::potentials::LJGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type =
      enable_tensor_view<Real>::scalar_type;

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::ljlk::potentials::LJGlobalParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol
