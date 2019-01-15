#pragma once

#include <cmath>
#include <tuple>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/cubic_hermite_polynomial.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using namespace tmol::score::common;
using std::tie;
using std::tuple;

template <typename Real>
struct LJTypeParams {
  Real lj_radius;
  Real lj_wdepth;
  bool is_donor;
  bool is_hydroxyl;
  bool is_polarh;
  bool is_acceptor;
};

template <typename Real>
struct LJGlobalParams {
  Real lj_hbond_dis;
  Real lj_hbond_OH_donor_dis;
  Real lj_hbond_hdis;
};

template <typename Real, typename Int>
Real connectivity_weight(Int bonded_path_length) {
  if (bonded_path_length > 4) {
    return 1.0;
  } else if (bonded_path_length == 4) {
    return 0.2;
  } else {
    return 0.0;
  }
}

template <typename Real>
Real lj_sigma(LJTypeParams<Real> i, LJTypeParams<Real> j, LJGlobalParams<Real> global) {
  if ((i.is_donor && !i.is_hydroxyl && j.is_acceptor)
      || (j.is_donor && !j.is_hydroxyl && i.is_acceptor)) {
    // standard donor/acceptor pair
    return global.lj_hbond_dis;
  } else if (
      (i.is_donor && i.is_hydroxyl && j.is_acceptor)
      || (j.is_donor && j.is_hydroxyl && i.is_acceptor)) {
    // hydroxyl donor/acceptor pair
    return global.lj_hbond_OH_donor_dis;
  } else if ((i.is_polarh && j.is_acceptor) || (j.is_polarh && i.is_acceptor)) {
    // hydrogen/acceptor pair
    return global.lj_hbond_hdis;
  } else {
    // standard lj
    return i.lj_radius + j.lj_radius;
  }
}

template <typename Real>
auto vdw_V(Real dist, Real sigma, Real epsilon) -> Real {
  Real sd = (sigma / dist);
  Real sd2 = sd * sd;
  Real sd6 = sd2 * sd2 * sd2;
  Real sd12 = sd6 * sd6;
  return epsilon * (sd12 - 2 * sd6);
}

template <typename Real>
auto vdw_V_dV(Real dist, Real sigma, Real epsilon) -> tuple<Real, Real> {
  Real sd = (sigma / dist);
  Real sd2 = sd * sd;
  Real sd6 = sd2 * sd2 * sd2;
  Real sd12 = sd6 * sd6;

  return {epsilon * (sd12 - 2.0 * sd6),
          epsilon * ((-12.0 * sd12 / dist) - (2.0 * -6.0 * sd6 / dist))};
}

template <typename Real>
auto lj_score_V(
    Real dist,
    Real bonded_path_length,
    LJTypeParams<Real> i,
    LJTypeParams<Real> j,
    LJGlobalParams<Real> global) -> Real {
  Real sigma = lj_sigma(i, j, global);
  Real weight = connectivity_weight<Real, Real>(bonded_path_length);
  Real epsilon = std::sqrt(i.lj_wdepth * j.lj_wdepth);

  Real d_lin = sigma * 0.6;
  Real cpoly_dmin = 4.5;
  Real cpoly_dmax = 6.0;

  if (dist < d_lin) {
    Real vdw, d_vdw_d_dist;
    tie(vdw, d_vdw_d_dist) = vdw_V_dV(d_lin, sigma, epsilon);
    return weight * (vdw + d_vdw_d_dist * (dist - d_lin));

  } else if (dist < cpoly_dmin) {
    Real vdw, d_vdw_d_dist;
    tie(vdw, d_vdw_d_dist) = vdw_V_dV(dist, sigma, epsilon);
    return weight * vdw;

  } else if (dist < cpoly_dmax) {
    Real vdw, d_vdw_d_dist;
    tie(vdw, d_vdw_d_dist) = vdw_V_dV(cpoly_dmin, sigma, epsilon);
    return weight
           * interpolate_to_zero(dist, cpoly_dmin, vdw, d_vdw_d_dist, cpoly_dmax);

  } else {
    return 0.0;
  }
}

template <typename Real>
auto lj_score_V_dV(
    Real dist,
    Real bonded_path_length,
    LJTypeParams<Real> i,
    LJTypeParams<Real> j,
    LJGlobalParams<Real> global) -> tuple<Real, Real> {
  Real sigma = lj_sigma(i, j, global);
  Real weight = connectivity_weight<Real, Real>(bonded_path_length);
  Real epsilon = std::sqrt(i.lj_wdepth * j.lj_wdepth);

  Real d_lin = sigma * 0.6;
  Real cpoly_dmin = 4.5;
  Real cpoly_dmax = 6.0;

  Real vdw, d_vdw_d_dist;
  Real lj, d_lj_d_dist;

  if (dist < d_lin) {
    tie(vdw, d_vdw_d_dist) = vdw_V_dV(d_lin, sigma, epsilon);

    lj = vdw + d_vdw_d_dist * (dist - d_lin);
    d_lj_d_dist = d_vdw_d_dist;

  } else if (dist < cpoly_dmin) {
    tie(vdw, d_vdw_d_dist) = vdw_V_dV(dist, sigma, epsilon);

    lj = vdw;
    d_lj_d_dist = d_vdw_d_dist;
  } else if (dist < cpoly_dmax) {
    tie(vdw, d_vdw_d_dist) = vdw_V_dV(cpoly_dmin, sigma, epsilon);
    tie(lj, d_lj_d_dist) =
        interpolate_to_zero_V_dV(dist, cpoly_dmin, vdw, d_vdw_d_dist, cpoly_dmax);

  } else {
    lj = 0.0;
    d_lj_d_dist = 0.0;
  }

  return {weight * lj, weight * d_lj_d_dist};
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
