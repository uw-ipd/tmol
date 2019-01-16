#include <pybind11/eigen.h>
#include <torch/torch.h>

#include "lj.hh"

using namespace tmol::score::ljlk::potentials;

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def("vdw_V_dV", &vdw_V_dV<Real>, "dist"_a, "sigma"_a, "epsilon"_a);
  m.def("vdw_V", &vdw_V<Real>, "dist"_a, "sigma"_a, "epsilon"_a);

  m.def(
      "lj_score_V",
      [](Real dist,
         Real bonded_path_length,

         Real i_lj_radius,
         Real i_lj_wdepth,
         bool i_is_donor,
         bool i_is_hydroxyl,
         bool i_is_polarh,
         bool i_is_acceptor,

         Real j_lj_radius,
         Real j_lj_wdepth,
         bool j_is_donor,
         bool j_is_hydroxyl,
         bool j_is_polarh,
         bool j_is_acceptor,

         Real lj_hbond_dis,
         Real lj_hbond_OH_donor_dis,
         Real lj_hbond_hdis) {
        return lj_score_V(
            dist,
            bonded_path_length,
            {i_lj_radius,
             i_lj_wdepth,
             i_is_donor,
             i_is_hydroxyl,
             i_is_polarh,
             i_is_acceptor},
            {j_lj_radius,
             j_lj_wdepth,
             j_is_donor,
             j_is_hydroxyl,
             j_is_polarh,
             j_is_acceptor},
            {lj_hbond_dis, lj_hbond_OH_donor_dis, lj_hbond_hdis});
      },
      "dist"_a,
      "bonded_path_length"_a,

      "i_lj_radius"_a,
      "i_lj_wdepth"_a,
      "i_is_donor"_a,
      "i_is_hydroxyl"_a,
      "i_is_polarh"_a,
      "i_is_acceptor"_a,

      "j_lj_radius"_a,
      "j_lj_wdepth"_a,
      "j_is_donor"_a,
      "j_is_hydroxyl"_a,
      "j_is_polarh"_a,
      "j_is_acceptor"_a,

      "lj_hbond_dis"_a,
      "lj_hbond_OH_donor_dis"_a,
      "lj_hbond_hdis"_a);

  m.def(
      "lj_score_V_dV",
      [](Real dist,
         Real bonded_path_length,

         Real i_lj_radius,
         Real i_lj_wdepth,
         bool i_is_donor,
         bool i_is_hydroxyl,
         bool i_is_polarh,
         bool i_is_acceptor,

         Real j_lj_radius,
         Real j_lj_wdepth,
         bool j_is_donor,
         bool j_is_hydroxyl,
         bool j_is_polarh,
         bool j_is_acceptor,

         Real lj_hbond_dis,
         Real lj_hbond_OH_donor_dis,
         Real lj_hbond_hdis) {
        return lj_score_V_dV(
            dist,
            bonded_path_length,
            {i_lj_radius,
             i_lj_wdepth,
             i_is_donor,
             i_is_hydroxyl,
             i_is_polarh,
             i_is_acceptor},
            {j_lj_radius,
             j_lj_wdepth,
             j_is_donor,
             j_is_hydroxyl,
             j_is_polarh,
             j_is_acceptor},
            {lj_hbond_dis, lj_hbond_OH_donor_dis, lj_hbond_hdis});
      },
      "dist"_a,
      "bonded_path_length"_a,

      "i_lj_radius"_a,
      "i_lj_wdepth"_a,
      "i_is_donor"_a,
      "i_is_hydroxyl"_a,
      "i_is_polarh"_a,
      "i_is_acceptor"_a,

      "j_lj_radius"_a,
      "j_lj_wdepth"_a,
      "j_is_donor"_a,
      "j_is_hydroxyl"_a,
      "j_is_polarh"_a,
      "j_is_acceptor"_a,

      "lj_hbond_dis"_a,
      "lj_hbond_OH_donor_dis"_a,
      "lj_hbond_hdis"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_potentials<float>(m);
  bind_potentials<double>(m);
}
