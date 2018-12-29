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
using std::tie;
using std::tuple;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>

struct AcceptorType {
  static constexpr int sp2 = 0;
  static constexpr int sp3 = 1;
  static constexpr int ring = 2;
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
  // In non-cos space
  auto [AHD, dAHD_dA, dAHD_dH, dAHD_dD] = pt_interior_angle_V_dV(A, H, D);
  auto [V, dV_dAHD] =
      bound_poly_V_dV(AHD, cosAHD_coeff, cosAHD_range, cosAHD_bound);

  return {V, dV_dAHD * dAHD_dA, dV_dAHD * dAHD_dH, dV_dAHD * dAHD_dD};
}

template <typename Real>
auto _BAH_angle_base_form_V_dV(
    Real3 B,
    Real3 A,
    Real3 H,
    Vec<Real, 11> cosBAH_coeff,
    Vec<Real, 2> cosBAH_range,
    Vec<Real, 2> cosBAH_bound) -> tuple<Real, Real3, Real3, Real3> {
  Real3 AH = H - A;
  Real3 BA = A - B;

  auto [cosT, d_cosT_dAH, d_cosT_dBA] = cos_interior_angle_V_dV(AH, BA);
  auto [V, dV_d_cosT] =
      bound_poly_V_dV(cosT, cosBAH_coeff, cosBAH_range, cosBAH_bound);

  return {V,
          dV_d_cosT * (-d_cosT_dBA),
          dV_d_cosT * (d_cosT_dBA - d_cosT_dAH),
          dV_d_cosT * d_cosT_dAH};
}

template <typename Real, typename Int>
auto BAH_angle_V_dV(
    Real3 B,
    Real3 B0,
    Real3 A,
    Real3 H,
    Int acceptor_type,
    Vec<Real, 11> cosBAH_coeff,
    Vec<Real, 2> cosBAH_range,
    Vec<Real, 2> cosBAH_bound,
    Real hb_sp3_softmax_fade) -> tuple<Real, Real3, Real3, Real3, Real3> {
  using std::exp;
  using std::log;

  if (acceptor_type == AcceptorType::sp2) {
    auto [PxH, dPxH_dB, dPxH_dA, dPxH_dH] = _BAH_angle_base_form_V_dV(
        B, A, H, cosBAH_coeff, cosBAH_range, cosBAH_bound);

    return {PxH, dPxH_dB, {0, 0, 0}, dPxH_dA, dPxH_dH};

  } else if (acceptor_type == AcceptorType::ring) {
    Real3 Bm = (B + B0) / 2;
    auto [PxHm, dPxH_dBm, dPxH_dA, dPxH_dH] = _BAH_angle_base_form_V_dV(
        Bm, A, H, cosBAH_coeff, cosBAH_range, cosBAH_bound);

    return {PxHm, dPxH_dBm / 2, dPxH_dBm / 2, dPxH_dA, dPxH_dH};

  } else if (acceptor_type == AcceptorType::sp3) {
    auto [PxH, dPxH_dB, dPxH_dA, dPxH_dH] = _BAH_angle_base_form_V_dV(
        B, A, H, cosBAH_coeff, cosBAH_range, cosBAH_bound);

    auto [PxH0, dPxH0_dB0, dPxH0_dA, dPxH0_dH] = _BAH_angle_base_form_V_dV(
        B0, A, H, cosBAH_coeff, cosBAH_range, cosBAH_bound);

    Real PxHfade =
        log(exp(PxH * hb_sp3_softmax_fade) + exp(PxH0 * hb_sp3_softmax_fade))
        / hb_sp3_softmax_fade;
    Real dPxHfade_dPxH =
        exp(PxH * hb_sp3_softmax_fade)
        / (exp(PxH * hb_sp3_softmax_fade) + exp(PxH0 * hb_sp3_softmax_fade));
    Real dPxHfade_dPxH0 =
        exp(PxH0 * hb_sp3_softmax_fade)
        / (exp(PxH * hb_sp3_softmax_fade) + exp(PxH0 * hb_sp3_softmax_fade));

    return {PxHfade,
            dPxHfade_dPxH * dPxH_dB,
            dPxHfade_dPxH0 * dPxH0_dB0,
            (dPxHfade_dPxH * dPxH_dA) + (dPxHfade_dPxH0 * dPxH0_dA),
            (dPxHfade_dPxH * dPxH_dH) + (dPxHfade_dPxH0 * dPxH0_dH)};
  } else {
    AT_ERROR("Invalid acceptor_type.");
  }
}

template <typename Real>
auto sp2chi_energy_V_dV(Real ang, Real chi, Real d, Real m, Real l)
    -> tuple<Real, Real, Real> {
  const Real pi = EIGEN_PI;

  using std::cos;
  using std::pow;
  using std::sin;

  Real H = 0.5 * (cos(2 * chi) + 1);

  if (ang > pi * 2.0 / 3.0) {
    Real F = d / 2 * cos(3 * (pi - ang)) + d / 2 - 0.5;
    Real G = d - 0.5;
    Real E = H * F + (1 - H) * G;

    Real dE_dang = 1.5 * d * sin(3 * ang) * cos(chi) * cos(chi);
    Real dE_dchi = 0.5 * d * (cos(3 * ang) + 1) * sin(2 * chi);

    return {E, dE_dang, dE_dchi};
  } else if (ang >= pi * (2.0 / 3.0 - l)) {
    Real outer_rise = cos(pi - (pi * 2 / 3 - ang) / l);

    Real F = m / 2 * outer_rise + m / 2 - 0.5;
    Real G = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5;

    Real E = H * F + (1 - H) * G;

    Real dE_dang =
        0.5 * (-d * sin(chi) * sin(chi) + m) * sin((ang - 2 * pi / 3) / l) / l;
    Real dE_dchi = 0.5 * d * (cos((ang - 2 * pi / 3) / l) + 1) * sin(2 * chi);

    return {E, dE_dang, dE_dchi};
  } else {
    // F = m - 0.5;
    // G = m - 0.5;

    Real E = m - 0.5;
    Real dE_dang = 0;
    Real dE_dchi = 0;

    return {E, dE_dang, dE_dchi};
  }
}

template <typename Real, typename Int>
auto B0BAH_chi_V_dV(
    Real3 B0,
    Real3 B,
    Real3 A,
    Real3 H,
    Int acceptor_type,
    Real hb_sp2_BAH180_rise,
    Real hb_sp2_range_span,
    Real hb_sp2_outer_width) -> tuple<Real, Real3, Real3, Real3, Real3> {
  if (acceptor_type == AcceptorType::sp2) {
    // SP-2 Chi Angle
    auto [BAH, dBAH_dB, dBAH_dA, dBAH_dH] = pt_interior_angle_V_dV(B, A, H);
    auto [B0BAH, dB0BAH_dB0, dB0BAH_dB, dB0BAH_dA, dB0BAH_dH] =
        dihedral_angle_V_dV(B0, B, A, H);

    auto [E, dE_dBAH, dE_dB0BAH] = sp2chi_energy_V_dV(
        BAH, B0BAH, hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width);

    return {
        E,
        dE_dB0BAH * dB0BAH_dB0,
        dE_dB0BAH * dB0BAH_dB + dE_dBAH * dBAH_dB,
        dE_dB0BAH * dB0BAH_dA + dE_dBAH * dBAH_dA,
        dE_dB0BAH * dB0BAH_dH + dE_dBAH * dBAH_dH,
    };
  } else {
    return {0, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  }
}

template <typename Real, typename Int>
Real hbond_score(
    // coordinates
    Vec<Real, 3> d,
    Vec<Real, 3> h,
    Vec<Real, 3> a,
    Vec<Real, 3> b,
    Vec<Real, 3> b0,

    // type pair parameters
    Int acceptor_type,
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

  // A-H Distance Component
  energy +=
      std::get<0>(AH_dist_V_dV(a, h, AHdist_coeff, AHdist_range, AHdist_bound));

  // AHD Angle Component
  energy += std::get<0>(
      AHD_angle_V_dV(a, h, d, cosAHD_coeff, cosAHD_range, cosAHD_bound));

  // BAH Angle Component
  energy += std::get<0>(BAH_angle_V_dV(
      b,
      b0,
      a,
      h,
      acceptor_type,
      cosBAH_coeff,
      cosBAH_range,
      cosBAH_bound,
      hb_sp3_softmax_fade));

  // B0BAH Chi Component
  energy += std::get<0>(B0BAH_chi_V_dV(
      b0,
      b,
      a,
      h,
      acceptor_type,
      hb_sp2_BAH180_rise,
      hb_sp2_range_span,
      hb_sp2_outer_width));

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
