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
struct LJLKTypeParams {
  Real lj_radius;
  Real lj_wdepth;
  Real lk_dgfree;
  Real lk_lambda;
  Real lk_volume;
  Real is_donor;
  Real is_hydroxyl;
  Real is_polarh;
  Real is_acceptor;

  LJTypeParams<Real> lj_params() {
    return LJTypeParams<Real>(
        {lj_radius, lj_wdepth, is_donor, is_hydroxyl, is_polarh, is_acceptor});
  }

  LKTypeParams<Real> lk_params() {
    return LKTypeParams<Real>({lj_radius,
                               lk_dgfree,
                               lk_lambda,
                               lk_volume,
                               is_donor,
                               is_hydroxyl,
                               is_polarh,
                               is_acceptor});
  }
};

template <typename Real>
struct LJGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};

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
    return LJTypeParams<Real>{lj_radius[i],
                              lj_wdepth[i],
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

  template <typename Idx>
  auto operator[](Idx i) const {
    return LKTypeParams<Real>{lj_radius[i],
                              lk_dgfree[i],
                              lk_lambda[i],
                              lk_volume[i],
                              is_donor[i],
                              is_hydroxyl[i],
                              is_polarh[i],
                              is_acceptor[i]};
  }
};

template <typename Real, tmol::Device D>
struct LJGlobalParamTensors {
  TView<Real, 1, D> lj_hbond_dis;
  TView<Real, 1, D> lj_hbond_OH_donor_dis;
  TView<Real, 1, D> lj_hbond_hdis;

  template <typename Idx>
  auto operator[](Idx i) const {
    return LJGlobalParams<Real>{
        lj_hbond_dis[i], lj_hbond_OH_donor_dis[i], lj_hbond_hdis[i]};
  }
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
