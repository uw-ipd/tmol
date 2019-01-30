#pragma once

#include <cmath>

#include <Eigen/Core>

#include <tmol/score/common/cubic_hermite_polynomial.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include "common.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define def                \
  template <typename Real> \
  auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

using namespace tmol::score::common;

def f_desolv_V(
    Real dist,
    Real lj_radius_i,
    Real lk_dgfree_i,
    Real lk_lambda_i,
    Real lk_volume_j)
    ->Real {
  using std::exp;
  using std::pow;
  const Real pi = EIGEN_PI;

  // clang-format off
  return (
      -lk_volume_j
      * lk_dgfree_i
      / (2 * pow(pi, 3.0 / 2.0) * lk_lambda_i)
      / (dist * dist)
      * exp(-pow((dist - lj_radius_i) / lk_lambda_i, 2))
  );
  // clang-format on
}

def f_desolv_V_dV(
    Real dist,
    Real lj_radius_i,
    Real lk_dgfree_i,
    Real lk_lambda_i,
    Real lk_volume_j)
    ->tuple<Real, Real> {
  using std::exp;
  using std::pow;
  const Real pi = EIGEN_PI;

  // clang-format off
  Real desolv = (
      -lk_volume_j
      * lk_dgfree_i
      / (2 * pow(pi, 3.0 / 2.0) * lk_lambda_i)
      / (dist * dist)
      * exp(-pow((dist - lj_radius_i) / lk_lambda_i, 2))
  );

  Real d_desolv_d_dist = (
      -lk_volume_j
      * lk_dgfree_i
      / (2 * pow(pi, 3.0 / 2.0) * lk_lambda_i)
      * ((  // (f * exp(g))' = f' * exp(g) + f g' exp(g)
          -2 / (dist * dist * dist)
          * exp(-pow(dist - lj_radius_i, 2) / pow(lk_lambda_i, 2))
        ) + (
          1 / (dist * dist)
          * -(2 * dist - 2 * lj_radius_i)
          / (lk_lambda_i * lk_lambda_i)
          * exp(-pow(dist - lj_radius_i, 2) / pow(lk_lambda_i, 2))
        )
      )
  );
  // clang-format on

  return {desolv, d_desolv_d_dist};
}

def lk_isotropic_pair_V(
    Real dist,
    Real bonded_path_length,
    Real lj_sigma_ij,
    Real lj_radius_i,
    Real lk_dgfree_i,
    Real lk_lambda_i,
    Real lk_volume_j)
    ->Real {
  Real d_min = lj_sigma_ij * .89;

  Real cpoly_close_dmin = std::sqrt(d_min * d_min - 1.45);
  Real cpoly_close_dmax = std::sqrt(d_min * d_min + 1.05);

  Real cpoly_far_dmin = 4.5;
  Real cpoly_far_dmax = 6.0;

  Real weight = connectivity_weight<Real>(bonded_path_length);

  Real lk;

  if (dist < cpoly_close_dmin) {
    lk = f_desolv_V(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

  } else if (dist < cpoly_close_dmax) {
    Real dmax_lk, dmax_lk_d_dist;
    tie(dmax_lk, dmax_lk_d_dist) = f_desolv_V_dV(
        cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

    lk = interpolate<Real>(
        dist,
        cpoly_close_dmin,
        f_desolv_V(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
        0.0,
        cpoly_close_dmax,
        dmax_lk,
        dmax_lk_d_dist);

  } else if (dist < cpoly_far_dmin) {
    lk = f_desolv_V(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

  } else if (dist < cpoly_far_dmax) {
    Real dmin_lk, dmin_lk_d_dist;
    tie(dmin_lk, dmin_lk_d_dist) = f_desolv_V_dV(
        cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

    lk = interpolate_to_zero(
        dist, cpoly_far_dmin, dmin_lk, dmin_lk_d_dist, cpoly_far_dmax);

  } else {
    lk = 0.0;
  }

  return weight * lk;
}

def lk_isotropic_pair_V_dV(
    Real dist,
    Real bonded_path_length,
    Real lj_sigma_ij,
    Real lj_radius_i,
    Real lk_dgfree_i,
    Real lk_lambda_i,
    Real lk_volume_j)
    ->tuple<Real, Real> {
  Real d_min = lj_sigma_ij * .89;

  Real cpoly_close_dmin = std::sqrt(d_min * d_min - 1.45);
  Real cpoly_close_dmax = std::sqrt(d_min * d_min + 1.05);

  Real cpoly_far_dmin = 4.5;
  Real cpoly_far_dmax = 6.0;

  Real weight = connectivity_weight<Real>(bonded_path_length);

  Real lk, d_lk_d_dist;

  if (dist < cpoly_close_dmin) {
    lk = f_desolv_V(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);
    d_lk_d_dist = 0;

  } else if (dist < cpoly_close_dmax) {
    Real dmax_lk, dmax_lk_d_dist;
    tie(dmax_lk, dmax_lk_d_dist) = f_desolv_V_dV(
        cpoly_close_dmax, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

    tie(lk, d_lk_d_dist) = interpolate_V_dV<Real>(
        dist,
        cpoly_close_dmin,
        f_desolv_V(d_min, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j),
        0.0,
        cpoly_close_dmax,
        dmax_lk,
        dmax_lk_d_dist);

  } else if (dist < cpoly_far_dmin) {
    tie(lk, d_lk_d_dist) =
        f_desolv_V_dV(dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

  } else if (dist < cpoly_far_dmax) {
    Real dmin_lk, dmin_lk_d_dist;
    tie(dmin_lk, dmin_lk_d_dist) = f_desolv_V_dV(
        cpoly_far_dmin, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j);

    tie(lk, d_lk_d_dist) = interpolate_to_zero_V_dV(
        dist, cpoly_far_dmin, dmin_lk, dmin_lk_d_dist, cpoly_far_dmax);

  } else {
    lk = 0.0;
    d_lk_d_dist = 0.0;
  }

  return {weight * lk, weight * d_lk_d_dist};
}

def lk_isotropic_score_V_dV(
    Real dist,
    Real bonded_path_length,
    LKTypeParams<Real> i,
    LKTypeParams<Real> j,
    LJGlobalParams<Real> global)
    ->tuple<Real, Real> {
  Real sigma = lj_sigma<Real>(i, j, global);

  return add(
      lk_isotropic_pair_V_dV(
          dist,
          bonded_path_length,
          sigma,
          i.lj_radius,
          i.lk_dgfree,
          i.lk_lambda,
          j.lk_volume),
      lk_isotropic_pair_V_dV(
          dist,
          bonded_path_length,
          sigma,
          j.lj_radius,
          j.lk_dgfree,
          j.lk_lambda,
          i.lk_volume));
}

def lk_isotropic_score_V(
    Real dist,
    Real bonded_path_length,
    LKTypeParams<Real> i,
    LKTypeParams<Real> j,
    LJGlobalParams<Real> global)
    ->Real {
  Real sigma = lj_sigma<Real>(i, j, global);

  return lk_isotropic_pair_V(
             dist,
             bonded_path_length,
             sigma,
             i.lj_radius,
             i.lk_dgfree,
             i.lk_lambda,
             j.lk_volume)
         + lk_isotropic_pair_V(
               dist,
               bonded_path_length,
               sigma,
               j.lj_radius,
               j.lk_dgfree,
               j.lk_lambda,
               i.lk_volume);
}

#undef def

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
