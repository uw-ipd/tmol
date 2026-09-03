#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/cubic_hermite_polynomial.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/polynomial.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#include "params.hh"

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

def connectivity_weight(Real bonded_path_length) -> Real {
  if (bonded_path_length > 4) {
    return Real(1);
  } else if (bonded_path_length == 4) {
    return Real(0.2);
  } else {
    return Real(0);
  }
}

// sigmoidal distance-dependant dielectric
def eps(Real dist, Real D, Real D0, Real S) -> Real {
  return (
      D
      - Real(0.5) * (D - D0)
            * (Real(2) + Real(2) * dist * S + dist * dist * S * S)
            * std::exp(-dist * S));
}

def deps_ddist(Real dist, Real D, Real D0, Real S) -> Real {
  return Real(0.5) * (D - D0) * dist * dist * S * S * S * std::exp(-dist * S);
}

def elec_delec_ddist(
    Real dist,
    Real e_i,
    Real e_j,
    Real bonded_path_length,
    ElecGlobalParams<Real> const& params) -> tuple<Real, Real> {
  Real low_poly_start = params.min_dis - Real(0.25);
  Real low_poly_end = params.min_dis + Real(0.25);
  Real hi_poly_start = params.max_dis - Real(1);
  Real hi_poly_end = params.max_dis;

  Real weight = connectivity_weight<Real>(bonded_path_length);

  Real eiej = e_i * e_j;

  Real elecE = 0, delec_ddist = 0;
  if (eiej == 0) {
    // Early exit for virtual atoms / atoms with a charge of 0
    return {elecE, delec_ddist};
  }

  if (dist < low_poly_start) {
    // flat part
    elecE = eiej * params.min_score;
    delec_ddist = 0;
  } else if (dist < low_poly_end) {
    // short range fade
    // Interesting thing to note here: If eiej is 0, you might
    // expect that interpolating between 0 and 0 would give you 0
    // everywhere, but it does NOT!
    tie(elecE, delec_ddist) = interpolate_V_dV<Real>(
        dist,
        low_poly_start,
        eiej * params.min_score,
        0.0,
        low_poly_end,
        eiej * params.low_score,
        eiej * params.low_deriv);

  } else if (dist < hi_poly_start) {
    // Coulombic part
    Real eps_elec = eps(dist, params.D, params.D0, params.S);
    Real deps_elec_d_dist = deps_ddist(dist, params.D, params.D0, params.S);

    elecE = eiej * (Real(322.0637) / (dist * eps_elec) - params.cutoff_offset);
    delec_ddist = -Real(322.0637) * eiej * (eps_elec + dist * deps_elec_d_dist)
                  / (dist * dist * eps_elec * eps_elec);

  } else if (dist < hi_poly_end) {
    // long range fade
    tie(elecE, delec_ddist) = interpolate_to_zero_V_dV(
        dist,
        hi_poly_start,
        eiej * params.high_score,
        eiej * params.high_deriv,
        hi_poly_end);
  }

  return {weight * elecE, weight * delec_ddist};
}

def elec(
    Real dist,
    Real e_i,
    Real e_j,
    Real bonded_path_length,
    ElecGlobalParams<Real> const& params) -> Real {
  Real low_poly_start = params.min_dis - Real(0.25);
  Real low_poly_end = params.min_dis + Real(0.25);
  Real hi_poly_start = params.max_dis - Real(1);
  Real hi_poly_end = params.max_dis;

  Real weight = connectivity_weight<Real>(bonded_path_length);

  Real eiej = e_i * e_j;
  if (eiej == 0) {
    // Early exit for virtual atoms / atoms with a charge of 0
    return 0;
  }

  Real elecE = 0;
  if (dist < low_poly_start) {
    // flat part
    elecE = eiej * params.min_score;
  } else if (dist < low_poly_end) {
    // short range fade
    // Interesting thing to note here: If eiej is 0, you might
    // expect that interpolating between 0 and 0 would give you 0
    // everywhere, but it does NOT!
    elecE = interpolate<Real>(
        dist,
        low_poly_start,
        eiej * params.min_score,
        0.0,
        low_poly_end,
        eiej * params.low_score,
        eiej * params.low_deriv);

  } else if (dist < hi_poly_start) {
    // Coulombic part
    Real eps_elec = eps(dist, params.D, params.D0, params.S);
    Real deps_elec_d_dist = deps_ddist(dist, params.D, params.D0, params.S);

    elecE = eiej * (Real(322.0637) / (dist * eps_elec) - params.cutoff_offset);

  } else if (dist < hi_poly_end) {
    // long range fade
    elecE = interpolate_to_zero(
        dist,
        hi_poly_start,
        eiej * params.high_score,
        eiej * params.high_deriv,
        hi_poly_end);
  }

  return weight * elecE;
}

#undef Real3
#undef def
}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
