#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "dispatch.hh"
#include "lj.hh"
#include "lk_isotropic.hh"

using namespace tmol::score::ljlk::potentials;

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "lj_sigma",
      py::vectorize([](LJTypeParams_args(i_),
                       LJTypeParams_args(j_),
                       LJGlobalParams_args()) {
        return lj_sigma<Real, LJTypeParams<Real>, LJGlobalParams<Real>>(
            LJTypeParams_struct(i_),
            LJTypeParams_struct(j_),
            LJGlobalParams_struct());
      }),
      LJTypeParams_pyargs(i_),
      LJTypeParams_pyargs(j_),
      LJGlobalParams_pyargs());

  m.def("vdw_V_dV", &vdw_V_dV<Real>, "dist"_a, "sigma"_a, "epsilon"_a);
  m.def("vdw_V", &vdw_V<Real>, "dist"_a, "sigma"_a, "epsilon"_a);

  m.def(
      "lj_score_V",
      [](Real dist,
         Real bonded_path_length,
         LJTypeParams_args(i_),
         LJTypeParams_args(j_),
         LJGlobalParams_args()) {
        return lj_score_V(
            dist,
            bonded_path_length,
            LJTypeParams_struct(i_),
            LJTypeParams_struct(j_),
            LJGlobalParams_struct());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LJTypeParams_pyargs(i_),
      LJTypeParams_pyargs(j_),
      LJGlobalParams_pyargs());

  m.def(
      "lj_score_V_dV",
      [](Real dist,
         Real bonded_path_length,
         LJTypeParams_args(i_),
         LJTypeParams_args(j_),
         LJGlobalParams_args()) {
        return lj_score_V_dV(
            dist,
            bonded_path_length,
            LJTypeParams_struct(i_),
            LJTypeParams_struct(j_),
            LJGlobalParams_struct());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LJTypeParams_pyargs(i_),
      LJTypeParams_pyargs(j_),
      LJGlobalParams_pyargs());

  m.def(
      "f_desolv_V",
      &f_desolv_V<Real>,
      "dist"_a,
      "lj_radius_i"_a,
      "lk_dgfree_i"_a,
      "lk_lambda_i"_a,
      "lk_volume_j"_a);

  m.def(
      "f_desolv_V_dV",
      &f_desolv_V_dV<Real>,
      "dist"_a,
      "lj_radius_i"_a,
      "lk_dgfree_i"_a,
      "lk_lambda_i"_a,
      "lk_volume_j"_a);

  m.def(
      "lk_isotropic_score_V",
      [](Real dist,
         Real bonded_path_length,
         LKTypeParams_args(i_),
         LKTypeParams_args(j_),
         LJGlobalParams_args()) {
        return lk_isotropic_score_V(
            dist,
            bonded_path_length,
            LKTypeParams_struct(i_),
            LKTypeParams_struct(j_),
            LJGlobalParams_struct());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LKTypeParams_pyargs(i_),
      LKTypeParams_pyargs(j_),
      LJGlobalParams_pyargs());

  m.def(
      "lk_isotropic_score_V_dV",
      [](Real dist,
         Real bonded_path_length,
         LKTypeParams_args(i_),
         LKTypeParams_args(j_),
         LJGlobalParams_args()) {
        return lk_isotropic_score_V_dV(
            dist,
            bonded_path_length,
            LKTypeParams_struct(i_),
            LKTypeParams_struct(j_),
            LJGlobalParams_struct());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LKTypeParams_pyargs(i_),
      LKTypeParams_pyargs(j_),
      LJGlobalParams_pyargs());
}

template <typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using tmol::score::common::NaiveDispatch;

  m.def(
      "lk_isotropic",
      &LKIsotropicDispatch<NaiveDispatch, tmol::Device::CPU, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LKTypeParams_pyargs(),
      LJGlobalParams_pyargs());

  m.def(
      "lk_isotropic_triu",
      &LKIsotropicDispatch<NaiveTriuDispatch, tmol::Device::CPU, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LKTypeParams_pyargs(),
      LJGlobalParams_pyargs());

  m.def(
      "lj",
      &LJDispatch<NaiveDispatch, tmol::Device::CPU, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LJTypeParams_pyargs(),
      LJGlobalParams_pyargs());

  m.def(
      "lj_triu",
      &LJDispatch<NaiveTriuDispatch, tmol::Device::CPU, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LJTypeParams_pyargs(),
      LJGlobalParams_pyargs());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_potentials<double>(m);
  bind_dispatch<float, int32_t>(m);
  bind_dispatch<float, int64_t>(m);
  bind_dispatch<double, int32_t>(m);
  bind_dispatch<double, int64_t>(m);
}
