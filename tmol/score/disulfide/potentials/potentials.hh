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

template <typename Real, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_disulfide_potential(
    const TensorAccessor<Vec<Real, 3>, 1, D> &coords,
    int block1_CA_ind,
    int block1_CB_ind,
    int block1_S_ind,
    int block2_S_ind,
    int block2_CB_ind,
    int block2_CA_ind,

    const DisulfideGlobalParams<Real> &params,

    Real &pose_V,
    TensorAccessor<Vec<Real, 3>, 1, D> pose_dV_dx) {
  auto block1_CA = coords[block1_CA_ind];
  auto block1_CB = coords[block1_CB_ind];
  auto block1_S = coords[block1_S_ind];

  auto block2_S = coords[block2_S_ind];
  auto block2_CB = coords[block2_CB_ind];
  auto block2_CA = coords[block2_CA_ind];

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
  const Real WT_DIH_SS(0.1);
  const Real WT_DIH_CS(0.1);
  const Real WT_ANG(0.1);
  const Real WT_LEN(0.1);
  const Real SHIFT(2.0);

  const Real res1_d_multiplier(1.0);
  const Real res2_d_multiplier(1.0);

  Real score = -SHIFT;

  {  // Calculate Distance
    // Score
    Real z = (ssdist.V - params.d_location) / params.d_scale;
    Real score_d =
        z * z / 2 - log(std::erfc(-params.d_shape * z / sqrt(2.0)) + MEST);
    score += WT_LEN * score_d;

    // Derivatives
    Real dscore_d =
        z / params.d_scale
        - (exp(-0.5 * z * z * params.d_shape * params.d_shape)
           * sqrt(2.0 / M_PI) * params.d_shape)
              / (params.d_scale * std::erfc(-params.d_shape * z / sqrt(2.0))
                 + 1.e-12);
    dscore_d *= WT_LEN;
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_S_ind], dscore_d * ssdist.dV_dA);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_S_ind], dscore_d * ssdist.dV_dB);
  }

  {  // Calculate Angles
    // Score
    Real ang_score(0);
    Real angle1(csang_1.V), angle2(csang_2.V);
    ang_score +=
        WT_ANG * (-params.a_logA - params.a_kappa * cos(angle1 - params.a_mu));
    ang_score +=
        WT_ANG * (-params.a_logA - params.a_kappa * cos(angle2 - params.a_mu));
    score += ang_score;

    // Derivatives
    Real dscore_a = params.a_kappa * sin(angle1 - params.a_mu) * WT_ANG;
    Real dscore_b = params.a_kappa * sin(angle2 - params.a_mu) * WT_ANG;
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_CB_ind], dscore_a * csang_1.dV_dA);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_S_ind], dscore_a * csang_1.dV_dB);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_S_ind], dscore_a * csang_1.dV_dC);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_CB_ind], dscore_b * csang_2.dV_dA);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_S_ind], dscore_b * csang_2.dV_dB);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_S_ind], dscore_b * csang_2.dV_dC);
  }

  {  // SS dihed
    // Score
    Real ang_ss(dihed.V), exp_score1(0.0), exp_score2(0.0);
    exp_score1 = exp(params.dss_logA1)
                 * exp(params.dss_kappa1
                       * cos(res1_d_multiplier * ang_ss - params.dss_mu1));
    exp_score2 = exp(params.dss_logA2)
                 * exp(params.dss_kappa2
                       * cos(res1_d_multiplier * ang_ss - params.dss_mu2));
    Real score_ss = -log(exp_score1 + exp_score2 + MEST);
    score += WT_DIH_SS * score_ss;

    // Derivatives
    Real dscore_ss(0.0);
    dscore_ss += exp_score1 * params.dss_kappa1
                 * sin(res1_d_multiplier * dihed.V - params.dss_mu1);
    dscore_ss += exp_score2 * params.dss_kappa2
                 * sin(res1_d_multiplier * dihed.V - params.dss_mu2);
    dscore_ss /= (exp_score1 + exp_score2 + MEST);
    dscore_ss *= WT_DIH_SS;

    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_CB_ind], dscore_ss * dihed.dV_dI);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_S_ind], dscore_ss * dihed.dV_dJ);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_S_ind], dscore_ss * dihed.dV_dK);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_CB_ind], dscore_ss * dihed.dV_dL);
  }

  {  // CB-S dihed
    // Score (angle 1)
    Real angle1(disulf_ca_dihedral_angle_1.V);
    Real exp_score1 = exp(params.dcs_logA1)
                      * exp(params.dcs_kappa1
                            * cos(res1_d_multiplier * angle1 - params.dcs_mu1));
    Real exp_score2 = exp(params.dcs_logA2)
                      * exp(params.dcs_kappa2
                            * cos(res1_d_multiplier * angle1 - params.dcs_mu2));
    Real exp_score3 = exp(params.dcs_logA3)
                      * exp(params.dcs_kappa3
                            * cos(res1_d_multiplier * angle1 - params.dcs_mu3));
    score += WT_DIH_CS * (-log(exp_score1 + exp_score2 + exp_score3 + MEST));

    // Derivatives (angle 2)
    Real dscore_cs = 0.0;
    dscore_cs += exp_score1 * params.dcs_kappa1 * sin(angle1 - params.dcs_mu1);
    dscore_cs += exp_score2 * params.dcs_kappa2 * sin(angle1 - params.dcs_mu2);
    dscore_cs += exp_score3 * params.dcs_kappa3 * sin(angle1 - params.dcs_mu3);
    dscore_cs /= (exp_score1 + exp_score2 + exp_score3 + MEST);
    dscore_cs *= WT_DIH_CS;

    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_CA_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dI);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_CB_ind],
        dscore_cs * disulf_ca_dihedral_angle_1.dV_dJ);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_S_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dK);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_S_ind], dscore_cs * disulf_ca_dihedral_angle_1.dV_dL);

    // Score (angle 2)
    Real angle2(disulf_ca_dihedral_angle_2.V);
    exp_score1 = exp(params.dcs_logA1)
                 * exp(params.dcs_kappa1
                       * cos(res2_d_multiplier * angle2 - params.dcs_mu1));
    exp_score2 = exp(params.dcs_logA2)
                 * exp(params.dcs_kappa2
                       * cos(res2_d_multiplier * angle2 - params.dcs_mu2));
    exp_score3 = exp(params.dcs_logA3)
                 * exp(params.dcs_kappa3
                       * cos(res2_d_multiplier * angle2 - params.dcs_mu3));
    score += WT_DIH_CS * (-log(exp_score1 + exp_score2 + exp_score3 + MEST));

    // Derivatives (angle 2)
    dscore_cs = 0.0;
    dscore_cs += exp_score1 * params.dcs_kappa1 * sin(angle2 - params.dcs_mu1);
    dscore_cs += exp_score2 * params.dcs_kappa2 * sin(angle2 - params.dcs_mu2);
    dscore_cs += exp_score3 * params.dcs_kappa3 * sin(angle2 - params.dcs_mu3);
    dscore_cs /= (exp_score1 + exp_score2 + exp_score3 + MEST);
    dscore_cs *= WT_DIH_CS;

    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_CA_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dI);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_CB_ind],
        dscore_cs * disulf_ca_dihedral_angle_2.dV_dJ);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block2_S_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dK);
    accumulate<D, Vec<Real, 3>>::add(
        pose_dV_dx[block1_S_ind], dscore_cs * disulf_ca_dihedral_angle_2.dV_dL);
  }

  accumulate<D, Real>::add(pose_V, score);
}

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
