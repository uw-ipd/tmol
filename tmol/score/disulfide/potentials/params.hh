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
  Real dss_kappa1;
  Real dss_mu1;
  Real dss_logA2;
  Real dss_kappa2;
  Real dss_mu2;

  Real dcs_logA1;
  Real dcs_mu1;
  Real dcs_kappa1;
  Real dcs_logA2;
  Real dcs_mu2;
  Real dcs_kappa2;
  Real dcs_logA3;
  Real dcs_mu3;
  Real dcs_kappa3;

  Real wt_dih_ss;
  Real wt_dih_cs;
  Real wt_ang;
  Real wt_len;
  Real shift;
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
