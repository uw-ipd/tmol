#pragma once

#include <cmath>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/polynomial.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using namespace tmol::score::common;
using std::tuple;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>

enum struct AcceptorType {
  sp2,
  sp3,
  ring,
};

template <typename Real>
auto AH_dist_V_dV(
    Real3 A,
    Real3 H,
    Vec<Real, 11> AHdist_coeff,
    Vec<Real, 2> AHdist_range,
    Vec<Real, 2> AHdist_bound) -> tuple<Real, Real3, Real3> {
  auto [D, dD_dA, dD_dH] = distance_V_dV(A, H);
  auto [V, dV_dD] =
      bound_poly_V_dV(D, AHdist_coeff, AHdist_range, AHdist_bound);

  return {V, dV_dD * dD_dA, dV_dD * dD_dH};
}

template <typename Real>
auto AHD_angle_V_dV(
    Real3 A,
    Real3 H,
    Real3 D,
    Vec<Real, 11> cosAHD_coeff,
    Vec<Real, 2> cosAHD_range,
    Vec<Real, 2> cosAHD_bound) -> tuple<Real, Real3, Real3, Real3> {
  auto [AHD, dAHD_dA, dAHD_dH, dAHD_dD] = pt_interior_angle_V_dV(A, H, D);
  auto [V, dV_dAHD] =
      bound_poly_V_dV(AHD, cosAHD_coeff, cosAHD_range, cosAHD_bound);

  return {V, dV_dAHD * dAHD_dA, dV_dAHD * dAHD_dH, dV_dAHD * dAHD_dD};
}

template <typename Real>
Real sp2chi_energy(Real d, Real m, Real l, Real BAH, Real chi) {
  const Real pi = EIGEN_PI;

  Real H = 0.5 * (std::cos(2 * chi) + 1);

  Real F;
  Real G;

  if (BAH > pi * 2.0 / 3.0) {
    F = d / 2 * std::cos(3 * (pi - BAH)) + d / 2 - 0.5;
    G = d - 0.5;
  } else if (BAH >= pi * (2.0 / 3.0 - l)) {
    Real outer_rise = std::cos(pi - (pi * 2 / 3 - BAH) / l);

    F = m / 2 * outer_rise + m / 2 - 0.5;
    G = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5;
  } else {
    F = m - 0.5;
    G = m - 0.5;
  }

  Real E = H * F + (1 - H) * G;

  return E;
}

template <typename Real>
Real hbond_score(
    // coordinates
    Vec<Real, 3> d,
    Vec<Real, 3> h,
    Vec<Real, 3> a,
    Vec<Real, 3> b,
    Vec<Real, 3> b0,

    // type pair parameters
    AcceptorType acceptor_type,
    Real glob_accwt,
    Real glob_donwt,

    Vec<Real, 11> AHdist_coeff,
    Vec<Real, 2> AHdist_range,
    Vec<Real, 2> AHdist_bound,

    Vec<Real, 11> cosBAH_coeff,
    Vec<Real, 2> cosBAH_range,
    Vec<Real, 2> cosBAH_bound,

    Vec<Real, 11> cosAHD_coeff,
    Vec<Real, 2> cosAHD_range,
    Vec<Real, 2> cosAHD_bound,

    // Global score parameters
    Real hb_sp2_range_span,
    Real hb_sp2_BAH180_rise,
    Real hb_sp2_outer_width,
    Real hb_sp3_softmax_fade) {
  const Real pi = EIGEN_PI;

  Real energy = 0.0;

  // Using Real3 nomenclature... xD = cos(pi - AHD); xH = cos(pi - BAH)

  // A-H Distance Component
  energy +=
      std::get<0>(AH_dist_V_dV(a, h, AHdist_coeff, AHdist_range, AHdist_bound));

  // AHD Angle Component
  energy += std::get<0>(
      AHD_angle_V_dV(a, h, d, cosAHD_coeff, cosAHD_range, cosAHD_bound));

  // BAH Angle Component
  // in cos space
  Real3 AH_unit_vec = (h - a).normalized();
  Real3 BA_base_b = (acceptor_type == AcceptorType::ring) ? 0.5 * (b + b0) : b;
  Real3 BA_unit_vec = (a - BA_base_b).normalized();

  Real xH = AH_unit_vec.dot(BA_unit_vec);
  Real PxH = bound_poly_V(xH, cosBAH_coeff, cosBAH_range, cosBAH_bound);

  if (acceptor_type == AcceptorType::sp3) {
    Real PxH1 = PxH;

    Real3 B0A_unit_vec = (a - b0).normalized();
    Real xH2 = AH_unit_vec.dot(B0A_unit_vec);
    Real PxH2 = bound_poly_V(xH2, cosBAH_coeff, cosBAH_range, cosBAH_bound);

    PxH =
        (1.0 / hb_sp3_softmax_fade
         * std::log(
               std::exp(PxH1 * hb_sp3_softmax_fade)
               + std::exp(PxH2 * hb_sp3_softmax_fade)));
  }

  energy += PxH;

  // SP-2 Chi Angle Component
  if (acceptor_type == AcceptorType::sp2) {
    Real BAH = pi - std::acos(xH);
    Real3 BB0_unit_vec = (b0 - b).normalized();
    Real xchi =
        BB0_unit_vec.dot(AH_unit_vec)
        - (BB0_unit_vec.dot(BA_unit_vec) * BA_unit_vec.dot(AH_unit_vec));

    Real ychi = BA_unit_vec.cross(AH_unit_vec).dot(BB0_unit_vec);
    Real chi = -std::atan2(ychi, xchi);

    Real Pchi = sp2chi_energy(
        hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width, BAH, chi);
    energy += Pchi;
  }

  // Donor/Acceptor Weighting
  energy *= glob_accwt * glob_donwt;

  // Truncate and Fade [-0.1,0.1] to [-0.1,0.0]
  if (energy > 0.1) {
    energy = 0.0;
  } else if (energy > -0.1) {
    energy = (-0.025 + 0.5 * energy - 2.5 * energy * energy);
  }

  return energy;
}

#undef Real3
}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
