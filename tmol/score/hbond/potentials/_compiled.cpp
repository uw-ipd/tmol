#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <tmol/score/hbond/potentials/dispatch.hh>
#include <tmol/score/hbond/potentials/potentials.hh>

using namespace tmol::score::hbond::potentials;

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
      "acceptor_class"_a,
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
      &hbond_score_V_dV<Real, int>,
      "HBond donor-acceptor geometry score.",

      "D"_a,
      "H"_a,
      "A"_a,
      "B"_a,
      "B0"_a,

      // type pair parameters
      "acceptor_class"_a,
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

template <typename Real>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "hbond_pair_score",
      &hbond_pair_score<Real, int32_t>,
      "D"_a,
      "H"_a,
      "donor_type"_a,

      "A"_a,
      "B"_a,
      "B0"_a,
      "acceptor_type"_a,

      "acceptor_class"_a,
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
      "hb_sp3_softmax_fade"_a,
      "threshold_distance"_a);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_potentials<double>(m);

  bind_dispatch<float>(m);
  bind_dispatch<double>(m);
}
