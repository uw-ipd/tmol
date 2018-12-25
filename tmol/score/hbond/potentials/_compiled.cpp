#include <pybind11/eigen.h>
#include <torch/torch.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <map>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <int N, typename Real>
using Vec = Eigen::Matrix<Real, N, 1>;

enum struct AcceptorType {
  sp2,
  sp3,
  ring,
};

template <int PDim, typename Real>
Real polyval(
    const Vec<PDim, Real>& coeffs,
    const Vec<2, Real>& range,
    const Vec<2, Real>& bound,
    const Real& x) {
  if (x < range[0]) {
    return bound[0];
  } else if (x > range[1]) {
    return bound[1];
  } else {
    Real p = coeffs(0);

#pragma unroll
    for (int i = 1; i < PDim; ++i) {
      p = p * x + coeffs[i];
    }

    return p;
  }
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

  // Using R3 nomenclature... xD = cos(pi - AHD); xH = cos(pi - BAH)

  // A-H Distance Component
  Real AHdist = (a - h).norm();
  Real P_AHdist = polyval(AHdist_coeff, AHdist_range, AHdist_bound, AHdist);
  energy += P_AHdist;

  // AHD Angle Component
  // in non-cos space
  Vec<3, Real> AH_unit_vec = (h - a).normalized();
  Vec<3, Real> HD_unit_vec = (d - h).normalized();

  Real xD = AH_unit_vec.dot(HD_unit_vec);
  Real AHD = pi - std::acos(xD);
  Real P_AHD = polyval(cosAHD_coeff, cosAHD_range, cosAHD_bound, AHD);
  energy += P_AHD;

  // BAH Angle Component
  // in cos space
  Vec<3, Real> BA_base_b =
      (acceptor_type == AcceptorType::ring) ? 0.5 * (b + b0) : b;
  Vec<3, Real> BA_unit_vec = (a - BA_base_b).normalized();

  Real xH = AH_unit_vec.dot(BA_unit_vec);
  Real PxH = polyval(cosBAH_coeff, cosBAH_range, cosBAH_bound, xH);

  if (acceptor_type == AcceptorType::sp3) {
    Real PxH1 = PxH;

    Vec<3, Real> B0A_unit_vec = (a - b0).normalized();
    Real xH2 = AH_unit_vec.dot(B0A_unit_vec);
    Real PxH2 = polyval(cosBAH_coeff, cosBAH_range, cosBAH_bound, xH2);

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

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "hbond_score",
      &hbond_score<Real>,
      "HBond donor-acceptor geometry score.",

      "d"_a,
      "h"_a,
      "a"_a,
      "b"_a,
      "b0"_a,

      // type pair parameters
      "acceptor_type"_a,
      "glob_accwt"_a,
      "glob_donwt"_a,

      "AHdist_coeff"_a,
      "AHdist_range"_a,
      "AHdist_bound"_a,

      "cosBAH_coeff"_a,
      "cosBAH_range"_a,
      "cosBAH_bound"_a,

      "cosAHD_coeff"_a,
      "cosAHD_range"_a,
      "cosAHD_bound"_a,

      // Global score parameters
      "hb_sp2_range_span"_a,
      "hb_sp2_BAH180_rise"_a,
      "hb_sp2_outer_width"_a,
      "hb_sp3_softmax_fade"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_potentials<float>(m);
  bind_potentials<double>(m);

  py::enum_<AcceptorType>(m, "AcceptorType")
      .value("sp2", AcceptorType::sp2)
      .value("sp3", AcceptorType::sp3)
      .value("ring", AcceptorType::ring);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
