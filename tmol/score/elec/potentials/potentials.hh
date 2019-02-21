#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/cubic_hermite_polynomial.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/polynomial.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#undef B0

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

#define def                \
  template <typename Real> \
  auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

using namespace tmol::score::common;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>

def connectivity_weight(Real bonded_path_length)->Real {
  if (bonded_path_length > 4) {
    return 1.0;
  } else if (bonded_path_length == 4) {
    return 0.2;
  } else {
    return 0.0;
  }
}

// sigmoidal distance-dependant dielectric
def eps(Real dist, Real D, Real D0, Real S)->Real {
  return (
      D
      - 0.5 * (D - D0) * (2 + 2 * dist * S + dist * dist * S * S)
            * std::exp(-dist * S));
}

def deps_ddist(Real dist, Real D, Real D0, Real S)->Real {
  return (0.5 * (D - D0) * dist * dist * S * S * S * std::exp(-dist * S));
}

def elec_delec_ddist(
    Real dist,
    Real e_i,
    Real e_j,
    Real bonded_path_length,
    Real D,
    Real D0,
    Real S,
    Real min_dis,
    Real max_dis)
    ->tuple<Real, Real> {
  Real low_poly_start = min_dis - 0.25;
  Real low_poly_end = min_dis + 0.25;
  Real hi_poly_start = max_dis - 1.0;
  Real hi_poly_end = max_dis;

  Real weight = connectivity_weight<Real>(bonded_path_length);

  Real C1 = 322.0637;  // electrostatic energy constant
  Real C2 = C1 / (max_dis * eps(max_dis, D, D0, S));

  Real eiej = e_i * e_j;

  Real elecE = 0, delec_ddist = 0;
  if (dist < low_poly_start) {
    // flat part
    Real min_dis_score = C1 / (min_dis * eps(min_dis, D, D0, S)) - C2;
    elecE = eiej * min_dis_score;
    delec_ddist = 0;
  } else if (dist < low_poly_end) {
    // short range fade
    Real min_dis_score = C1 / (min_dis * eps(min_dis, D, D0, S)) - C2;
    Real eps_elec = eps(low_poly_end, D, D0, S);
    Real deps_elec_d_dist = deps_ddist(low_poly_end, D, D0, S);
    Real dmax_elec = eiej * (C1 / (low_poly_end * eps_elec) - C2);
    Real dmax_elec_d_dist =
        -C1 * eiej * (eps_elec + low_poly_end * deps_elec_d_dist)
        / (low_poly_end * low_poly_end * eps_elec * eps_elec);

    tie(elecE, delec_ddist) = interpolate_V_dV<Real>(
        dist,
        low_poly_start,
        eiej * min_dis_score,
        0.0,
        low_poly_end,
        dmax_elec,
        deps_elec_d_dist);

  } else if (dist < hi_poly_start) {
    // Coulombic part
    Real eps_elec = eps(dist, D, D0, S);
    Real deps_elec_d_dist = deps_ddist(dist, D, D0, S);

    elecE = eiej * (C1 / (dist * eps_elec) - C2);
    delec_ddist = -C1 * eiej * (eps_elec + dist * deps_elec_d_dist)
                  / (dist * dist * eps_elec * eps_elec);

  } else if (dist < hi_poly_end) {
    // long range fade
    Real eps_elec = eps(hi_poly_start, D, D0, S);
    Real deps_elec_d_dist = deps_ddist(hi_poly_start, D, D0, S);
    Real dmin_elec = eiej * (C1 / (hi_poly_start * eps_elec) - C2);
    Real dmin_elec_d_dist =
        -C1 * eiej * (eps_elec + hi_poly_start * deps_elec_d_dist)
        / (hi_poly_start * hi_poly_start * eps_elec * eps_elec);

    tie(elecE, delec_ddist) = interpolate_to_zero_V_dV(
        dist, hi_poly_start, dmin_elec, dmin_elec_d_dist, hi_poly_end);
  }

  return {weight * elecE, weight * delec_ddist};
}

#undef Real3
#undef def
}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
