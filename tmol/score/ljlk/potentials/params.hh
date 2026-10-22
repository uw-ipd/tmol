#pragma once

#include <tmol/utility/tensor/native.hh>
#include <cmath>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include "params_types.hh"

namespace tmol {

template <typename Real>
struct enable_tensor_view<score::ljlk::potentials::LJTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

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
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::ljlk::potentials::LKTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::ljlk::potentials::LJLKTypeParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::ljlk::potentials::LJLKTypeParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

template <typename Real>
struct enable_tensor_view<score::ljlk::potentials::LJGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const tmol::native::ScalarType scalar_type() {
    return enable_tensor_view<Real>::scalar_type();
  }

  static const int nconsumed_dims = 1;
  static const int consumed_dims(int i) {
    return (i == 0) ? sizeof(score::ljlk::potentials::LJGlobalParams<Real>)
                          / sizeof(Real)
                    : 0;
  }

  typedef typename enable_tensor_view<Real>::PrimitiveType PrimitiveType;
};

}  // namespace tmol

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

// Views over the per-type parameter tables.
//
// These live here rather than beside the POD parameter structs in
// params_types.hh because they hold TViews, and params_types.hh is included
// by device code that the Metal toolchain compiles -- which cannot see the
// torch tensor layer at all. The kernels read the POD structs; only the host
// builds these.

template <typename Real, tmol::Device D>
struct LJTypeParamTensors {
  TView<Real, 1, D> lj_radius;
  TView<Real, 1, D> lj_wdepth;
  TView<bool, 1, D> is_donor;
  TView<bool, 1, D> is_hydroxyl;
  TView<bool, 1, D> is_polarh;
  TView<bool, 1, D> is_acceptor;

  template <typename Idx>
  auto operator[](Idx i) const {
    return LJTypeParams<Real>{
        lj_radius[i],
        std::sqrt(lj_wdepth[i]),
        is_donor[i],
        is_hydroxyl[i],
        is_polarh[i],
        is_acceptor[i]};
  }
};

template <typename Real, tmol::Device D>
struct LKTypeParamTensors {
  TView<Real, 1, D> lj_radius;
  TView<Real, 1, D> lk_dgfree;
  TView<Real, 1, D> lk_lambda;
  TView<Real, 1, D> lk_volume;
  TView<bool, 1, D> is_donor;
  TView<bool, 1, D> is_hydroxyl;
  TView<bool, 1, D> is_polarh;
  TView<bool, 1, D> is_acceptor;
  TView<bool, 1, D> is_carbon_lk;

  template <typename Idx>
  auto operator[](Idx i) const {
    return LKTypeParams<Real>{
        lj_radius[i],
        lk_dgfree[i],
        lk_lambda[i],
        lk_volume[i],
        is_donor[i],
        is_hydroxyl[i],
        is_polarh[i],
        is_acceptor[i],
        is_carbon_lk[i],
        -lk_dgfree[i] / (Real(2) * Real(5.56832799683) * lk_lambda[i]),
        Real(1) / (lk_lambda[i] * lk_lambda[i])};
  }
};

template <typename Real, tmol::Device D>
struct LJGlobalParamTensors {
  TView<Real, 1, D> max_dis;
  TView<Real, 1, D> lj_dlin_sigma_factor;
  TView<Real, 1, D> lj_hbond_dis;
  TView<Real, 1, D> lj_hbond_OH_donor_dis;
  TView<Real, 1, D> lj_hbond_hdis;

  template <typename Idx>
  auto operator[](Idx i) const {
    return LJGlobalParams<Real>{
        max_dis[i],
        lj_dlin_sigma_factor[i],
        lj_hbond_dis[i],
        lj_hbond_OH_donor_dis[i],
        lj_hbond_hdis[i]};
  }
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
