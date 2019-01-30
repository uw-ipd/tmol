#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_potentials<double>(m); }

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
