#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/extension.h>

#include <tmol/score/hbond/potentials/potentials.hh>

namespace pybind11 {
namespace detail {

using tmol::score::hbond::potentials::hbond_score_V_dV_t;
using tmol::score::hbond::potentials::HBondGlobalParams;
using tmol::score::hbond::potentials::HBondPairParams;
using tmol::score::hbond::potentials::HBondPoly;
using tmol::score::hbond::potentials::HBondPolynomials;
using tmol::score::hbond::potentials::Vec;

template <typename Real>
struct type_caster<hbond_score_V_dV_t<Real>> {
 public:
  PYBIND11_TYPE_CASTER(hbond_score_V_dV_t<Real>, _("hbond_score_V_dV_t"));

  static handle cast(
      hbond_score_V_dV_t<Real> src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::cast(std::make_tuple(
                        src.V,
                        src.dV_dD,
                        src.dV_dH,
                        src.dV_dA,
                        src.dV_dB,
                        src.dV_dB0))
        .release();
  }
};

template <typename Real>
struct type_caster<HBondPoly<Real>> {
  typedef HBondPoly<Real> T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool) {
    Vec<Real, 16> vals = src.cast<Vec<Real, 16>>();
    for (int ii = 0; ii < 11; ++ii) {
      value.coeffs[ii] = vals[ii];
    }
    for (int ii = 0; ii < 2; ++ii) {
      value.range[ii] = vals[ii + 12];
      value.bound[ii] = vals[ii + 14];
    }
    return true;
  }
};

template <typename Real>
struct type_caster<HBondPolynomials<Real>> {
  typedef HBondPolynomials<Real> T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool) {
    Vec<Real, 48> vals = src.cast<Vec<Real, 48>>();
    for (int ii = 0; ii < 11; ++ii) {
      value.AHdist_poly.coeffs[ii] = vals[ii];
      value.cosBAH_poly.coeffs[ii] = vals[ii + 16];
      value.cosAHD_poly.coeffs[ii] = vals[ii + 32];
    }
    for (int ii = 0; ii < 2; ++ii) {
      value.AHdist_poly.range[ii] = vals[ii + 12];
      value.AHdist_poly.bound[ii] = vals[ii + 14];
      value.cosBAH_poly.range[ii] = vals[ii + 12 + 16];
      value.cosBAH_poly.bound[ii] = vals[ii + 14 + 16];
      value.cosAHD_poly.range[ii] = vals[ii + 12 + 32];
      value.cosAHD_poly.bound[ii] = vals[ii + 14 + 32];
    }
    return true;
  }
};

template <typename Real>
struct type_caster<HBondGlobalParams<Real>> {
  typedef HBondGlobalParams<Real> T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool) {
    Vec<Real, 5> vals = src.cast<Vec<Real, 5>>();
    value.hb_sp2_range_span = vals[0];
    value.hb_sp2_BAH180_rise = vals[1];
    value.hb_sp2_outer_width = vals[2];
    value.hb_sp3_softmax_fade = vals[3];
    value.threshold_distance = vals[4];
    return true;
  }
};

template <typename Real>
struct type_caster<HBondPairParams<Real>> {
  typedef HBondPairParams<Real> T;
  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool) {
    Vec<Real, 3> vals = src.cast<Vec<Real, 3>>();
    value.acceptor_hybridization = vals[0];
    value.acceptor_weight = vals[1];
    value.donor_weight = vals[2];
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def("AH_dist_V_dV", &AH_dist_V_dV<Real>, "A"_a, "H"_a, "AHdist_poly"_a);

  m.def(
      "AHD_angle_V_dV",
      &AHD_angle_V_dV<Real>,
      "A"_a,
      "H"_a,
      "D"_a,
      "cosAHD_poly"_a);

  m.def(
      "BAH_angle_V_dV",
      &BAH_angle_V_dV<Real, int>,
      "B"_a,
      "B0"_a,
      "A"_a,
      "H"_a,
      "acceptor_hybridization"_a,
      "cosBAH_poly"_a,
      "hb_sp3_softmax_fade"_a);

  m.def(
      "sp2chi_energy_V_dV",
      &sp2chi_energy_V_dV<Real>,
      "BAH_angle"_a,
      "B0BAH_chi"_a,
      "hb_sp2_BAH180_rise"_a,
      "hb_sp2_range_span"_a,
      "hb_sp2_outer_width"_a);

  m.def(
      "hbond_score_V_dV",
      &hbond_score<Real, int>::V_dV,
      "HBond donor-acceptor geometry score.",

      "D"_a,
      "H"_a,
      "A"_a,
      "B"_a,
      "B0"_a,

      // type pair parameters
      "pair_params"_a,
      "polynomials"_a,
      "global_params"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_potentials<double>(m); }
}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
