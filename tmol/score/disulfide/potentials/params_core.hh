#pragma once

#include <type_traits>

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

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
  Real dss_mixed_logA1;
  Real dss_mixed_kappa1;
  Real dss_mixed_mu1;
  Real dss_mixed_logA2;
  Real dss_mixed_kappa2;
  Real dss_mixed_mu2;
  Real chirality;
};

static_assert(std::is_standard_layout_v<DisulfideGlobalParams<float>>);
static_assert(std::is_standard_layout_v<DisulfideGlobalParams<double>>);
static_assert(sizeof(DisulfideGlobalParams<float>) == 33 * sizeof(float));
static_assert(sizeof(DisulfideGlobalParams<double>) == 33 * sizeof(double));

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
