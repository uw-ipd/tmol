#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pybind11/pybind11.h>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

using namespace tmol::score::common;

template <typename Real>
struct DihedralScore {
  Real V;
  Real dV;
};

template <typename Real, int N>
TMOL_DEVICE_FUNC DihedralScore<Real> dihedral_mixture(
    Real angle,
    const Real (&logA)[N],
    const Real (&kappa)[N],
    const Real (&mu)[N]) {
  Real density = exp(Real(-20));
  Real derivative = 0;
  for (int i = 0; i < N; ++i) {
    Real delta = angle - mu[i];
    Real component = exp(logA[i] + kappa[i] * cos(delta));
    density += component;
    derivative += component * kappa[i] * sin(delta);
  }
  return {-log(density), derivative / density};
}

template <typename Real>
TMOL_DEVICE_FUNC DihedralScore<Real> ss_dihedral(
    Real angle,
    const DisulfideGlobalParams<Real>& p1,
    const DisulfideGlobalParams<Real>& p2) {
  // The mixed distribution is symmetric in both chirality order and angle.
  // Homochiral pairs retain the existing (mirrored for DD) parameter row.
  bool mixed = p1.chirality != p2.chirality;
  Real logA[] = {
      mixed ? p1.dss_mixed_logA1 : p1.dss_logA1,
      mixed ? p1.dss_mixed_logA2 : p1.dss_logA2};
  Real kappa[] = {
      mixed ? p1.dss_mixed_kappa1 : p1.dss_kappa1,
      mixed ? p1.dss_mixed_kappa2 : p1.dss_kappa2};
  Real mu[] = {
      mixed ? p1.dss_mixed_mu1 : p1.dss_mu1,
      mixed ? p1.dss_mixed_mu2 : p1.dss_mu2};
  return dihedral_mixture(angle, logA, kappa, mu);
}

template <typename Real>
TMOL_DEVICE_FUNC DihedralScore<Real> cs_dihedral(
    Real angle, const DisulfideGlobalParams<Real>& p) {
  Real logA[] = {p.dcs_logA1, p.dcs_logA2, p.dcs_logA3};
  Real kappa[] = {p.dcs_kappa1, p.dcs_kappa2, p.dcs_kappa3};
  Real mu[] = {p.dcs_mu1, p.dcs_mu2, p.dcs_mu3};
  return dihedral_mixture(angle, logA, kappa, mu);
}

template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_disulfide(
    TView<Vec<Real, 3>, 1, D> coords,
    int A_CA_ind,
    int A_CB_ind,
    int A_S_ind,
    int B_S_ind,
    int B_CB_ind,
    int B_CA_ind,
    // params1 and params2 are the parameters of the two residues. They differ
    // only in the sign of the dihedral means: reflecting a residue negates the
    // dihedrals it contributes, and cos is even, so negating the mean is the
    // same operation and needs no correction to the derivative.
    const DisulfideGlobalParams<Real>& params1,
    const DisulfideGlobalParams<Real>& params2,
    TView<Vec<Real, 3>, 2, D> dV_dx,
    // The backward pass scales the derivatives by dT/dV and wants no score;
    // the forward pass wants the score and scales by one.
    Real dTdV,
    Real* V) {
  auto A_CA = coords[A_CA_ind];
  auto A_CB = coords[A_CB_ind];
  auto A_S = coords[A_S_ind];

  auto B_S = coords[B_S_ind];
  auto B_CB = coords[B_CB_ind];
  auto B_CA = coords[B_CA_ind];

  auto ssdist = distance<Real>::V_dV(A_S, B_S);
  auto csang_1 = pt_interior_angle<Real>::V_dV(A_CB, A_S, B_S);
  auto csang_2 = pt_interior_angle<Real>::V_dV(B_CB, B_S, A_S);
  auto dihed = dihedral_angle<Real>::V_dV(A_CB, A_S, B_S, B_CB);
  auto disulf_ca_dihedral_angle_1 =
      dihedral_angle<Real>::V_dV(A_CA, A_CB, A_S, B_S);
  auto disulf_ca_dihedral_angle_2 =
      dihedral_angle<Real>::V_dV(B_CA, B_CB, B_S, A_S);

  const Real MEST = exp(-20.0);

  Real score = -params1.shift;

  {  // Calculate Distance
    // Score
    Real z = (ssdist.V - params1.d_location) / params1.d_scale;
    Real score_d =
        z * z / 2 - log(std::erfc(-params1.d_shape * z / sqrt(2.0)) + MEST);
    score += params1.wt_len * score_d;

    // Derivatives
    Real dscore_d =
        z / params1.d_scale
        - (exp(-0.5 * z * z * params1.d_shape * params1.d_shape)
           * sqrt(2.0 / M_PI) * params1.d_shape)
              / (params1.d_scale
                 * (std::erfc(-params1.d_shape * z / sqrt(2.0)) + MEST));
    dscore_d *= params1.wt_len;
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_S_ind], dscore_d * ssdist.dV_dA * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_S_ind], dscore_d * ssdist.dV_dB * dTdV);
  }

  {  // Calculate Angles
    // Score
    Real ang_score(0);
    Real angle1(csang_1.V), angle2(csang_2.V);
    ang_score +=
        params1.wt_ang
        * (-params1.a_logA - params1.a_kappa * cos(angle1 - params1.a_mu));
    ang_score +=
        params1.wt_ang
        * (-params1.a_logA - params1.a_kappa * cos(angle2 - params1.a_mu));
    score += ang_score;

    // Derivatives
    Real dscore_a =
        params1.a_kappa * sin(angle1 - params1.a_mu) * params1.wt_ang;
    Real dscore_b =
        params1.a_kappa * sin(angle2 - params1.a_mu) * params1.wt_ang;
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_CB_ind], dscore_a * csang_1.dV_dA * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_S_ind], dscore_a * csang_1.dV_dB * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_S_ind], dscore_a * csang_1.dV_dC * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_CB_ind], dscore_b * csang_2.dV_dA * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_S_ind], dscore_b * csang_2.dV_dB * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_S_ind], dscore_b * csang_2.dV_dC * dTdV);
  }

  {  // SS dihed
    auto ss = ss_dihedral(dihed.V, params1, params2);
    score += params1.wt_dih_ss * ss.V;
    Real dscore_ss = params1.wt_dih_ss * ss.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_CB_ind], dscore_ss * dihed.dV_dI * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_S_ind], dscore_ss * dihed.dV_dJ * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_S_ind], dscore_ss * dihed.dV_dK * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_CB_ind], dscore_ss * dihed.dV_dL * dTdV);
  }

  {  // CB-S dihed
    auto cs1 = cs_dihedral(disulf_ca_dihedral_angle_1.V, params1);
    auto cs2 = cs_dihedral(disulf_ca_dihedral_angle_2.V, params2);
    score += params1.wt_dih_cs * cs1.V + params2.wt_dih_cs * cs2.V;
    Real dscore_cs = params1.wt_dih_cs * cs1.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_CA_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dI * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_CB_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dJ * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_S_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dK * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_S_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dL * dTdV);

    dscore_cs = params2.wt_dih_cs * cs2.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_CA_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dI * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_CB_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dJ * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][B_S_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dK * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][A_S_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dL * dTdV);
  }

  if (V != nullptr) {
    // Atomic even for block-pair scoring: rotamer-pair scoring assigns one
    // output thread per disulfide connection, and a block type could carry
    // more than one of them.
    accumulate<D, Real>::add(*V, score);
  }
}

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
