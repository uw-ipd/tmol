#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/score/common/geom.hh>
#include <tmol/score/common/polynomial.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/tuple_operators.hh>

#undef B0

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using namespace tmol::score::common;

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>

struct AcceptorHybridization {
  static constexpr int none = 0;
  static constexpr int sp2 = 1;
  static constexpr int sp3 = 2;
  static constexpr int ring = 3;
};

template <typename Real>
def AH_dist_V_dV(
    Real3 A,
    Real3 H,
    Vec<double, 11> AHdist_coeffs,
    Vec<double, 2> AHdist_range,
    Vec<double, 2> AHdist_bound)
    ->tuple<Real, Real3, Real3> {
  auto dist = distance<Real>::V_dV(A, H);
  auto poly = bound_poly<11, double>::V_dV(
      dist.V, AHdist_coeffs, AHdist_range, AHdist_bound);

  return {poly.V, poly.dV_dX * dist.dV_dA, poly.dV_dX * dist.dV_dB};
}

template <typename Real>
def AHD_angle_V_dV(
    Real3 A,
    Real3 H,
    Real3 D,
    Vec<double, 11> cosAHD_coeffs,
    Vec<double, 2> cosAHD_range,
    Vec<double, 2> cosAHD_bound)
    ->tuple<Real, Real3, Real3, Real3> {
  // In non-cos space
  auto AHD = pt_interior_angle<Real>::V_dV(A, H, D);
  auto poly = bound_poly<11, double>::V_dV(
      AHD.V, cosAHD_coeffs, cosAHD_range, cosAHD_bound);

  return {poly.V,
          poly.dV_dX * AHD.dV_dA,
          poly.dV_dX * AHD.dV_dB,
          poly.dV_dX * AHD.dV_dC};
}

template <typename Real>
def _BAH_angle_base_form_V_dV(
    Real3 B,
    Real3 A,
    Real3 H,
    Vec<double, 11> cosBAH_coeffs,
    Vec<double, 2> cosBAH_range,
    Vec<double, 2> cosBAH_bound)
    ->tuple<Real, Real3, Real3, Real3> {
  Real3 AH = H - A;
  Real3 BA = A - B;

  auto cosT = cos_interior_angle<Real>::V_dV(AH, BA);
  auto poly = bound_poly<11, double>::V_dV(
      cosT.V, cosBAH_coeffs, cosBAH_range, cosBAH_bound);

  return {poly.V,
          poly.dV_dX * (-cosT.dV_dB),
          poly.dV_dX * (cosT.dV_dB - cosT.dV_dA),
          poly.dV_dX * cosT.dV_dA};
}

template <typename Real, typename Int>
def BAH_angle_V_dV(
    Real3 B,
    Real3 B0,
    Real3 A,
    Real3 H,
    Int acceptor_hybridization,
    Vec<double, 11> cosBAH_coeffs,
    Vec<double, 2> cosBAH_range,
    Vec<double, 2> cosBAH_bound,
    Real hb_sp3_softmax_fade)
    ->tuple<Real, Real3, Real3, Real3, Real3> {
  using std::exp;
  using std::log;

  if (acceptor_hybridization == AcceptorHybridization::sp2) {
    Real PxH;
    Real3 dPxH_dB, dPxH_dA, dPxH_dH;
    tie(PxH, dPxH_dB, dPxH_dA, dPxH_dH) = _BAH_angle_base_form_V_dV(
        B, A, H, cosBAH_coeffs, cosBAH_range, cosBAH_bound);

    return {PxH, dPxH_dB, Real3({0, 0, 0}), dPxH_dA, dPxH_dH};

  } else if (acceptor_hybridization == AcceptorHybridization::ring) {
    Real3 Bm = (B + B0) / 2;
    Real PxHm;
    Real3 dPxH_dBm, dPxH_dA, dPxH_dH;
    tie(PxHm, dPxH_dBm, dPxH_dA, dPxH_dH) = _BAH_angle_base_form_V_dV(
        Bm, A, H, cosBAH_coeffs, cosBAH_range, cosBAH_bound);

    return {PxHm, dPxH_dBm / 2, dPxH_dBm / 2, dPxH_dA, dPxH_dH};

  } else if (acceptor_hybridization == AcceptorHybridization::sp3) {
    Real PxH;
    Real3 dPxH_dB, dPxH_dA, dPxH_dH;
    tie(PxH, dPxH_dB, dPxH_dA, dPxH_dH) = _BAH_angle_base_form_V_dV(
        B, A, H, cosBAH_coeffs, cosBAH_range, cosBAH_bound);

    Real PxH0;
    Real3 dPxH0_dB0, dPxH0_dA, dPxH0_dH;
    tie(PxH0, dPxH0_dB0, dPxH0_dA, dPxH0_dH) = _BAH_angle_base_form_V_dV(
        B0, A, H, cosBAH_coeffs, cosBAH_range, cosBAH_bound);

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
#ifndef __CUDACC__
    throw std::runtime_error("Invalid acceptor_hybridization.");
#endif
  }
}

template <typename Real>
def sp2chi_energy_V_dV(Real ang, Real chi, Real d, Real m, Real l)
    ->tuple<Real, Real, Real> {
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
def B0BAH_chi_V_dV(
    Real3 B0,
    Real3 B,
    Real3 A,
    Real3 H,
    Int acceptor_hybridization,
    Real hb_sp2_BAH180_rise,
    Real hb_sp2_range_span,
    Real hb_sp2_outer_width)
    ->tuple<Real, Real3, Real3, Real3, Real3> {
  if (acceptor_hybridization == AcceptorHybridization::sp2) {
    // SP-2 Chi Angle
    Real BAH;
    Real3 dBAH_dB, dBAH_dA, dBAH_dH;
    tie(BAH, dBAH_dB, dBAH_dA, dBAH_dH) =
        pt_interior_angle<Real>::V_dV(B, A, H).astuple();

    Real B0BAH;
    Real3 dB0BAH_dB0, dB0BAH_dB, dB0BAH_dA, dB0BAH_dH;
    tie(B0BAH, dB0BAH_dB0, dB0BAH_dB, dB0BAH_dA, dB0BAH_dH) =
        dihedral_angle<Real>::V_dV(B0, B, A, H).astuple();

    Real E, dE_dBAH, dE_dB0BAH;
    tie(E, dE_dBAH, dE_dB0BAH) = sp2chi_energy_V_dV(
        BAH, B0BAH, hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width);

    return {
        E,
        dE_dB0BAH * dB0BAH_dB0,
        dE_dB0BAH * dB0BAH_dB + dE_dBAH * dBAH_dB,
        dE_dB0BAH * dB0BAH_dA + dE_dBAH * dBAH_dA,
        dE_dB0BAH * dB0BAH_dH + dE_dBAH * dBAH_dH,
    };
  } else {
    return {0,
            Real3({0, 0, 0}),
            Real3({0, 0, 0}),
            Real3({0, 0, 0}),
            Real3({0, 0, 0})};
  }
}

template <typename Real>
struct hbond_score_V_dV_t {
  Real V;

  Real3 dV_dD;
  Real3 dV_dH;

  Real3 dV_dA;
  Real3 dV_dB;
  Real3 dV_dB0;
};

template <typename Real, typename Int>
struct hbond_score {
  static def V_dV(
      // coordinates
      Real3 D,
      Real3 H,
      Real3 A,
      Real3 B,
      Real3 B0,

      // type pair parameters
      Int acceptor_hybridization,
      Real acceptor_weight,
      Real donor_weight,

      Vec<double, 11> AHdist_coeffs,
      Vec<double, 2> AHdist_range,
      Vec<double, 2> AHdist_bound,

      Vec<double, 11> cosBAH_coeffs,
      Vec<double, 2> cosBAH_range,
      Vec<double, 2> cosBAH_bound,

      Vec<double, 11> cosAHD_coeffs,
      Vec<double, 2> cosAHD_range,
      Vec<double, 2> cosAHD_bound,

      // Global score parameters
      Real hb_sp2_range_span,
      Real hb_sp2_BAH180_rise,
      Real hb_sp2_outer_width,
      Real hb_sp3_softmax_fade)
      ->hbond_score_V_dV_t<Real> {
    Real E = 0.0;
    Real3 dE_dD = {0, 0, 0};
    Real3 dE_dH = {0, 0, 0};
    Real3 dE_dA = {0, 0, 0};
    Real3 dE_dB = {0, 0, 0};
    Real3 dE_dB0 = {0, 0, 0};

    // A-H Distance Component
    iadd(
        tie(E, dE_dA, dE_dH),
        AH_dist_V_dV(A, H, AHdist_coeffs, AHdist_range, AHdist_bound));

    // AHD Angle Component
    iadd(
        tie(E, dE_dA, dE_dH, dE_dD),
        AHD_angle_V_dV(A, H, D, cosAHD_coeffs, cosAHD_range, cosAHD_bound));

    // BAH Angle Component
    iadd(
        tie(E, dE_dB, dE_dB0, dE_dA, dE_dH),
        BAH_angle_V_dV(
            B,
            B0,
            A,
            H,
            acceptor_hybridization,
            cosBAH_coeffs,
            cosBAH_range,
            cosBAH_bound,
            hb_sp3_softmax_fade));

    // B0BAH Chi Component
    iadd(
        tie(E, dE_dB0, dE_dB, dE_dA, dE_dH),
        B0BAH_chi_V_dV(
            B0,
            B,
            A,
            H,
            acceptor_hybridization,
            hb_sp2_BAH180_rise,
            hb_sp2_range_span,
            hb_sp2_outer_width));

    // Donor/Acceptor Weighting
    E *= acceptor_weight * donor_weight;
    dE_dD *= acceptor_weight * donor_weight;
    dE_dH *= acceptor_weight * donor_weight;
    dE_dA *= acceptor_weight * donor_weight;
    dE_dB *= acceptor_weight * donor_weight;
    dE_dB0 *= acceptor_weight * donor_weight;

    // Truncate and Fade [-0.1,0.1] to [-0.1,0.0]
    if (E > 0.1) {
      E = 0;
      dE_dD = {0, 0, 0};
      dE_dH = {0, 0, 0};
      dE_dA = {0, 0, 0};
      dE_dB = {0, 0, 0};
      dE_dB0 = {0, 0, 0};

    } else if (E > -0.1) {
      E = (-0.025 + 0.5 * E - 2.5 * E * E);
      dE_dD *= -5.0 * E + 0.5;
      dE_dH *= -5.0 * E + 0.5;
      dE_dA *= -5.0 * E + 0.5;
      dE_dB *= -5.0 * E + 0.5;
      dE_dB0 *= -5.0 * E + 0.5;
    }

    return {E, dE_dD, dE_dH, dE_dA, dE_dB, dE_dB0};
  }
};

#undef Real3
#undef def
}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
