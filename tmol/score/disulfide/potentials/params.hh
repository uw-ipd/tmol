#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template <typename T, int N>
using Vec = Eigen::Matrix<T, N, 1>;

template <typename Real>
struct DisulfideGlobalParams {
  Real d_location;
  Real d_scale;
  Real d_shape;

  Real a_logA;
  Real a_kappa;
  Real a_mu;

  Real dss_logA1;
  Real dss_mixed_logA1;
  Real dss_kappa1;
  Real dss_mixed_kappa1;
  Real dss_mu1;
  Real dss_mixed_mu1;
  Real dss_logA2;
  Real dss_mixed_logA2;
  Real dss_kappa2;
  Real dss_mixed_kappa2;
  Real dss_mu2;
  Real dss_mixed_mu2;

  Real dcs_logA1;
  Real dcs_mu1;
  Real dcs_kappa1;
  Real dcs_logA2;
  Real dcs_mu2;
  Real dcs_kappa2;
  Real dcs_logA3;
  Real dcs_mu3;
  Real dcs_kappa3;
};

template <typename Real, tmol::Device D>
struct DisulfideGlobalParamTensors {
  TView<Real, 1, D> d_location;
  TView<Real, 1, D> d_scale;
  TView<Real, 1, D> d_shape;

  TView<Real, 1, D> a_logA;
  TView<Real, 1, D> a_kappa;
  TView<Real, 1, D> a_mu;

  TView<Real, 1, D> dss_logA1;
  TView<Real, 1, D> dss_mixed_logA1;
  TView<Real, 1, D> dss_kappa1;
  TView<Real, 1, D> dss_mixed_kappa1;
  TView<Real, 1, D> dss_mu1;
  TView<Real, 1, D> dss_mixed_mu1;
  TView<Real, 1, D> dss_logA2;
  TView<Real, 1, D> dss_mixed_logA2;
  TView<Real, 1, D> dss_kappa2;
  TView<Real, 1, D> dss_mixed_kappa2;
  TView<Real, 1, D> dss_mu2;
  TView<Real, 1, D> dss_mixed_mu2;

  TView<Real, 1, D> dcs_logA1;
  TView<Real, 1, D> dcs_mu1;
  TView<Real, 1, D> dcs_kappa1;
  TView<Real, 1, D> dcs_logA2;
  TView<Real, 1, D> dcs_mu2;
  TView<Real, 1, D> dcs_kappa2;
  TView<Real, 1, D> dcs_logA3;
  TView<Real, 1, D> dcs_mu3;
  TView<Real, 1, D> dcs_kappa3;

  template <typename Idx>
  auto operator[](Idx i) const {
    return DisulfideGlobalParams<Real>{
        d_location[i], d_scale[i],          d_shape[i],

        a_logA[i],     a_kappa[i],          a_mu[i],

        dss_logA1[i],  dss_mixed_logA1[i],  dss_kappa1[i], dss_mixed_kappa1[i],
        dss_mu1[i],    dss_mixed_mu1[i],    dss_logA2[i],  dss_mixed_logA2[i],
        dss_kappa2[i], dss_mixed_kappa2[i], dss_mu2[i],    dss_mixed_mu2[i],

        dcs_logA1[i],  dcs_mu1[i],          dcs_kappa1[i], dcs_logA2[i],
        dcs_mu2[i],    dcs_kappa2[i],       dcs_logA3[i],  dcs_mu3[i],
        dcs_kappa3[i]};
  }
};

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol

namespace tmol {

template <typename Real>
struct enable_tensor_view<
    score::disulfide::potentials::DisulfideGlobalParams<Real>> {
  static const bool enabled = enable_tensor_view<Real>::enabled;
  static const at::ScalarType scalar_type() {
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
