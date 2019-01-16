#pragma once

#include <cmath>
#include <tuple>

#include <tmol/score/common/cubic_hermite_polynomial.hh>

#include "common.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using namespace tmol::score::common;
using std::tie;
using std::tuple;

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
  Real sigma = lj_sigma<Real>(i, j, global);
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
  Real sigma = lj_sigma<Real>(i, j, global);
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
