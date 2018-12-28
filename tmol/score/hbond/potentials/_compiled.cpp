#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <tmol/score/hbond/potentials/compiled.hh>

using namespace tmol::score::hbond::potentials;

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "AH_dist_v",
      &AH_dist_v<Real>,
      "h"_a,
      "a"_a,
      "AHdist_coeff"_a,
      "AHdist_range"_a,
      "AHdist_bound"_a);

  m.def(
      "AH_dist_v_d",
      &AH_dist_v_d<Real>,
      "h"_a,
      "a"_a,
      "AHdist_coeff"_a,
      "AHdist_range"_a,
      "AHdist_bound"_a);

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
