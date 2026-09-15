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
TMOL_DEVICE_FUNC void accumulate_disulfide_potential(
    TView<Vec<Real, 3>, 1, D> rot_coords,
    int pose_ind,
    int rot1_ind,
    int rot1_CA_ind,

    int rot1_CB_ind,
    int rot1_S_ind,
    int rot2_ind,
    int rot2_S_ind,
    int rot2_CB_ind,

    int rot2_CA_ind,
    // params1 and params2 are the parameters of the two residues. They differ
    // only in the sign of the dihedral means: reflecting a residue negates the
    // dihedrals it contributes, and cos is even, so negating the mean is the
    // same operation and needs no correction to the derivative.
    const DisulfideGlobalParams<Real>& params1,
    const DisulfideGlobalParams<Real>& params2,
    bool output_block_pair_energies,
    Real& V,
    TView<Vec<Real, 3>, 2, D> dV_dx) {
  auto rot1_CA = rot_coords[rot1_CA_ind];
  auto rot1_CB = rot_coords[rot1_CB_ind];
  auto rot1_S = rot_coords[rot1_S_ind];

  auto rot2_S = rot_coords[rot2_S_ind];
  auto rot2_CB = rot_coords[rot2_CB_ind];
  auto rot2_CA = rot_coords[rot2_CA_ind];

  auto ssdist = distance<Real>::V_dV(rot1_S, rot2_S);
  auto csang_1 = pt_interior_angle<Real>::V_dV(rot1_CB, rot1_S, rot2_S);
  auto csang_2 = pt_interior_angle<Real>::V_dV(rot2_CB, rot2_S, rot1_S);
  auto dihed = dihedral_angle<Real>::V_dV(rot1_CB, rot1_S, rot2_S, rot2_CB);
  auto disulf_ca_dihedral_angle_1 =
      dihedral_angle<Real>::V_dV(rot1_CA, rot1_CB, rot1_S, rot2_S);
  auto disulf_ca_dihedral_angle_2 =
      dihedral_angle<Real>::V_dV(rot2_CA, rot2_CB, rot2_S, rot1_S);

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
        dV_dx[0][rot1_S_ind], dscore_d * ssdist.dV_dA);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_S_ind], dscore_d * ssdist.dV_dB);
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
        dV_dx[0][rot1_CB_ind], dscore_a * csang_1.dV_dA);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_S_ind], dscore_a * csang_1.dV_dB);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_S_ind], dscore_a * csang_1.dV_dC);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_CB_ind], dscore_b * csang_2.dV_dA);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_S_ind], dscore_b * csang_2.dV_dB);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_S_ind], dscore_b * csang_2.dV_dC);
  }

  {  // SS dihed
    auto ss = ss_dihedral(dihed.V, params1, params2);
    score += params1.wt_dih_ss * ss.V;
    Real dscore_ss = params1.wt_dih_ss * ss.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_CB_ind], dscore_ss * dihed.dV_dI);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_S_ind], dscore_ss * dihed.dV_dJ);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_S_ind], dscore_ss * dihed.dV_dK);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_CB_ind], dscore_ss * dihed.dV_dL);
  }

  {  // CB-S dihed
    auto cs1 = cs_dihedral(disulf_ca_dihedral_angle_1.V, params1);
    auto cs2 = cs_dihedral(disulf_ca_dihedral_angle_2.V, params2);
    score += params1.wt_dih_cs * cs1.V + params2.wt_dih_cs * cs2.V;
    Real dscore_cs = params1.wt_dih_cs * cs1.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_CA_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dI);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_CB_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dJ);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_S_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dK);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_S_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dL);

    dscore_cs = params2.wt_dih_cs * cs2.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_CA_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dI);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_CB_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dJ);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot2_S_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dK);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][rot1_S_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dL);
  }

  if (output_block_pair_energies) {
    // Note that we must still use atomic increment here
    // because, even though in block-pair scoring we only
    // assign a single thread to each block pair, in
    // rotamer-pair scoring, we assign one output thread
    // per disulfide connection, which, you could possibly
    // imagine there being more than one of in a single block
    // type (some hypothetical di-cysteine non-canonical AA)
    accumulate<D, Real>::add(V, score);
  } else {
    accumulate<D, Real>::add(V, score);
  }
}

template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_disulfide_derivs(
    TView<Vec<Real, 3>, 1, D> rot_coords,
    int block1_ind,
    int block1_CA_ind,
    int block1_CB_ind,
    int block1_S_ind,
    int block2_ind,
    int block2_S_ind,
    int block2_CB_ind,
    int block2_CA_ind,

    // params1 and params2 are the parameters of the two residues. They differ
    // only in the sign of the dihedral means: reflecting a residue negates the
    // dihedrals it contributes, and cos is even, so negating the mean is the
    // same operation and needs no correction to the derivative.
    const DisulfideGlobalParams<Real>& params1,
    const DisulfideGlobalParams<Real>& params2,

    TView<Vec<Real, 3>, 2, D> dV_dx,
    Real dTdV) {
  //   Real block_weight =
  //       0.5 * (dTdV[block1_ind][block2_ind] + dTdV[block2_ind][block1_ind]);

  auto block1_CA = rot_coords[block1_CA_ind];
  auto block1_CB = rot_coords[block1_CB_ind];
  auto block1_S = rot_coords[block1_S_ind];

  auto block2_S = rot_coords[block2_S_ind];
  auto block2_CB = rot_coords[block2_CB_ind];
  auto block2_CA = rot_coords[block2_CA_ind];

  auto ssdist = distance<Real>::V_dV(block1_S, block2_S);
  auto csang_1 = pt_interior_angle<Real>::V_dV(block1_CB, block1_S, block2_S);
  auto csang_2 = pt_interior_angle<Real>::V_dV(block2_CB, block2_S, block1_S);
  auto dihed =
      dihedral_angle<Real>::V_dV(block1_CB, block1_S, block2_S, block2_CB);
  auto disulf_ca_dihedral_angle_1 =
      dihedral_angle<Real>::V_dV(block1_CA, block1_CB, block1_S, block2_S);
  auto disulf_ca_dihedral_angle_2 =
      dihedral_angle<Real>::V_dV(block2_CA, block2_CB, block2_S, block1_S);

  const Real MEST = exp(-20.0);

  Real score = -params1.shift;

  {  // Calculate Distance
    // Derivatives
    Real z = (ssdist.V - params1.d_location) / params1.d_scale;
    Real dscore_d =
        z / params1.d_scale
        - (exp(-0.5 * z * z * params1.d_shape * params1.d_shape)
           * sqrt(2.0 / M_PI) * params1.d_shape)
              / (params1.d_scale
                 * (std::erfc(-params1.d_shape * z / sqrt(2.0)) + MEST));
    dscore_d *= params1.wt_len;
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_S_ind], dscore_d * ssdist.dV_dA * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_S_ind], dscore_d * ssdist.dV_dB * dTdV);
  }

  {  // Calculate Angles
    // Derivatives
    Real angle1(csang_1.V), angle2(csang_2.V);
    Real dscore_a =
        params1.a_kappa * sin(angle1 - params1.a_mu) * params1.wt_ang;
    Real dscore_b =
        params1.a_kappa * sin(angle2 - params1.a_mu) * params1.wt_ang;
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_CB_ind], dscore_a * csang_1.dV_dA * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_S_ind], dscore_a * csang_1.dV_dB * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_S_ind], dscore_a * csang_1.dV_dC * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_CB_ind], dscore_b * csang_2.dV_dA * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_S_ind], dscore_b * csang_2.dV_dB * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_S_ind], dscore_b * csang_2.dV_dC * dTdV);
  }

  {  // SS dihed
    auto ss = ss_dihedral(dihed.V, params1, params2);
    Real dscore_ss = params1.wt_dih_ss * ss.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_CB_ind], dscore_ss * dihed.dV_dI * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_S_ind], dscore_ss * dihed.dV_dJ * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_S_ind], dscore_ss * dihed.dV_dK * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_CB_ind], dscore_ss * dihed.dV_dL * dTdV);
  }

  {  // CB-S dihed
    auto cs1 = cs_dihedral(disulf_ca_dihedral_angle_1.V, params1);
    auto cs2 = cs_dihedral(disulf_ca_dihedral_angle_2.V, params2);
    Real dscore_cs = params1.wt_dih_cs * cs1.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_CA_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dI * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_CB_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dJ * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_S_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dK * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_S_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dL * dTdV);

    dscore_cs = params2.wt_dih_cs * cs2.dV;

    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_CA_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dI * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_CB_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dJ * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block2_S_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dK * dTdV);
    accumulate<D, Vec<Real, 3>>::add(
        dV_dx[0][block1_S_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dL * dTdV);
  }
}

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
