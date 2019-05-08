#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/extension.h>

#include <tmol/score/hbond/potentials/potentials.hh>

namespace pybind11 {
namespace detail {

using tmol::score::hbond::potentials::hbond_score_V_dV_t;
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
}  // namespace detail
}  // namespace pybind11

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "AH_dist_V_dV",
      &AH_dist_V_dV<Real>,
      "A"_a,
      "H"_a,
      "AHdist_coeffs"_a,
      "AHdist_range"_a,
      "AHdist_bound"_a);

  m.def(
      "AHD_angle_V_dV",
      &AHD_angle_V_dV<Real>,
      "A"_a,
      "H"_a,
      "D"_a,
      "cosAHD_coeffs"_a,
      "cosAHD_range"_a,
      "cosAHD_bound"_a);

  m.def(
      "BAH_angle_V_dV",
      &BAH_angle_V_dV<Real, int>,
      "B"_a,
      "B0"_a,
      "A"_a,
      "H"_a,
      "acceptor_hybridization"_a,
      "cosBAH_coeffs"_a,
      "cosBAH_range"_a,
      "cosBAH_bound"_a,
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
      "acceptor_hybridization"_a,
      "acceptor_weight"_a,
      "donor_weight"_a,

      "AHdist_coeffs"_a,
      "AHdist_range"_a,
      "AHdist_bound"_a,

      "cosBAH_coeffs"_a,
      "cosBAH_range"_a,
      "cosBAH_bound"_a,

      "cosAHD_coeffs"_a,
      "cosAHD_range"_a,
      "cosAHD_bound"_a,

      // Global score parameters
      "hb_sp2_range_span"_a,
      "hb_sp2_BAH180_rise"_a,
      "hb_sp2_outer_width"_a,
      "hb_sp3_softmax_fade"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_potentials<double>(m); }
}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
