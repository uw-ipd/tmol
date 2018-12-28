#pragma once

#include <cmath>
#include <tuple>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/polynomial.hh>
#include <tmol/score/common/vec.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {
using namespace tmol::score::common;

enum struct AcceptorType {
  sp2,
  sp3,
  ring,
};

template <int PDim, typename Real>
Real bound_poly_v(
    const Real& x,
    const Vec<PDim, Real>& coeffs,
    const Vec<2, Real>& range,
    const Vec<2, Real>& bound) {
  if (x < range[0]) {
    return bound[0];
  } else if (x > range[1]) {
    return bound[1];
  } else {
    return poly_v(x, coeffs);
  }
}

template <int PDim, typename Real>
std::tuple<Real, Real> bound_poly_v_d(
    const Real& x,
    const Vec<PDim, Real>& coeffs,
    const Vec<2, Real>& range,
    const Vec<2, Real>& bound) {
  if (x < range[0]) {
    return {bound[0], 0};
  } else if (x > range[1]) {
    return {bound[1], 0};
  } else {
    return poly_v_d(x, coeffs);
  }
}

template <typename Real>
Real AH_dist_v(
    const Vec<3, Real>& h,
    const Vec<3, Real>& a,
    const Vec<11, Real>& AHdist_coeff,
    const Vec<2, Real>& AHdist_range,
    const Vec<2, Real>& AHdist_bound) {
  return bound_poly_v((a - h).norm(), AHdist_coeff, AHdist_range, AHdist_bound);
}

template <typename Real>
std::tuple<Real, Vec<3, Real>, Vec<3, Real>> AH_dist_v_d(
    const Vec<3, Real>& h,
    const Vec<3, Real>& a,
    const Vec<11, Real>& AHdist_coeff,
    const Vec<2, Real>& AHdist_range,
    const Vec<2, Real>& AHdist_bound) {
  Real n = (a - h).norm();

  auto [v, d_v_d_n] =
      bound_poly_v_d(n, AHdist_coeff, AHdist_range, AHdist_bound);

  Vec<3, Real> d_v_d_h = d_v_d_n * (h - a) / n;
  Vec<3, Real> d_v_d_a = d_v_d_n * (a - h) / n;

  return {v, d_v_d_h, d_v_d_a};
}

template <typename Real>
Real sp2chi_energy(
    const Real& d,
    const Real& m,
    const Real& l,
    const Real& BAH,
    const Real& chi) {
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
    const Vec<3, Real>& d,
    const Vec<3, Real>& h,
    const Vec<3, Real>& a,
    const Vec<3, Real>& b,
    const Vec<3, Real>& b0,

    // type pair parameters
    const AcceptorType& acceptor_type,
    const Real& glob_accwt,
    const Real& glob_donwt,

    const Vec<11, Real>& AHdist_coeff,
    const Vec<2, Real>& AHdist_range,
    const Vec<2, Real>& AHdist_bound,

    const Vec<11, Real>& cosBAH_coeff,
    const Vec<2, Real>& cosBAH_range,
    const Vec<2, Real>& cosBAH_bound,

    const Vec<11, Real>& cosAHD_coeff,
    const Vec<2, Real>& cosAHD_range,
    const Vec<2, Real>& cosAHD_bound,

    // Global score parameters
    const Real& hb_sp2_range_span,
    const Real& hb_sp2_BAH180_rise,
    const Real& hb_sp2_outer_width,
    const Real& hb_sp3_softmax_fade) {
  const Real pi = EIGEN_PI;

  Real energy = 0.0;

  // Using Real3 nomenclature... xD = cos(pi - AHD); xH = cos(pi - BAH)

  // A-H Distance Component
  energy += AH_dist_v(a, h, AHdist_coeff, AHdist_range, AHdist_bound);

  // AHD Angle Component
  // in non-cos space
  Vec<3, Real> AH_unit_vec = (h - a).normalized();
  Vec<3, Real> HD_unit_vec = (d - h).normalized();

  Real xD = AH_unit_vec.dot(HD_unit_vec);
  Real AHD = pi - std::acos(xD);
  Real P_AHD = bound_poly_v(AHD, cosAHD_coeff, cosAHD_range, cosAHD_bound);
  energy += P_AHD;

  // BAH Angle Component
  // in cos space
  Vec<3, Real> BA_base_b =
      (acceptor_type == AcceptorType::ring) ? 0.5 * (b + b0) : b;
  Vec<3, Real> BA_unit_vec = (a - BA_base_b).normalized();

  Real xH = AH_unit_vec.dot(BA_unit_vec);
  Real PxH = bound_poly_v(xH, cosBAH_coeff, cosBAH_range, cosBAH_bound);

  if (acceptor_type == AcceptorType::sp3) {
    Real PxH1 = PxH;

    Vec<3, Real> B0A_unit_vec = (a - b0).normalized();
    Real xH2 = AH_unit_vec.dot(B0A_unit_vec);
    Real PxH2 = bound_poly_v(xH2, cosBAH_coeff, cosBAH_range, cosBAH_bound);

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
    Vec<3, Real> BB0_unit_vec = (b0 - b).normalized();
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

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
