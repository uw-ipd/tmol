#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>
#include <tmol/score/ljlk/potentials/params.pybind.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "lj_sigma",
      &lj_sigma<Real, LJTypeParams<Real>, LJGlobalParams<Real>>,
      "i_params"_a,
      "j_params"_a,
      "global_params"_a);

  m.def(
      "vdw_V_dV",
      [](Real dist, Real sigma, Real epsilon) {
        return vdw<Real>::V_dV(dist, sigma, epsilon).astuple();
      },
      "dist"_a,
      "sigma"_a,
      "epsilon"_a);
  m.def("vdw_V", &vdw<Real>::V, "dist"_a, "sigma"_a, "epsilon"_a);

  m.def(
      "lj_score_V",
      &lj_score<Real>::V,
      "dist"_a,
      "bonded_path_length"_a,
      "i_params"_a,
      "j_params"_a,
      "global_params"_a);

  m.def(
      "lj_score_V_dV",
      [](Real dist,
         Real bonded_path_length,
         LJTypeParams<Real> i,
         LJTypeParams<Real> j,
         LJGlobalParams<Real> global) {
        return lj_score<Real>::V_dV(dist, bonded_path_length, i, j, global)
            .astuple();
      },
      "dist"_a,
      "bonded_path_length"_a,
      "i_params"_a,
      "j_params"_a,
      "global_params"_a);

  m.def(
      "f_desolv_V",
      &f_desolv<Real>::V,
      "dist"_a,
      "lj_radius_i"_a,
      "lk_dgfree_i"_a,
      "lk_lambda_i"_a,
      "lk_volume_j"_a);

  m.def(
      "f_desolv_V_dV",
      [](

          Real dist,
          Real lj_radius_i,
          Real lk_dgfree_i,
          Real lk_lambda_i,
          Real lk_volume_j) {
        return f_desolv<Real>::V_dV(
                   dist, lj_radius_i, lk_dgfree_i, lk_lambda_i, lk_volume_j)
            .astuple();
      },
      "dist"_a,
      "lj_radius_i"_a,
      "lk_dgfree_i"_a,
      "lk_lambda_i"_a,
      "lk_volume_j"_a);

  m.def(
      "lk_isotropic_score_V",
      &lk_isotropic_score<Real>::V,
      "dist"_a,
      "bonded_path_length"_a,
      "i"_a,
      "j"_a,
      "global"_a);

  m.def(
      "lk_isotropic_score_V_dV",
      [](Real dist,
         Real bonded_path_length,
         LKTypeParams<Real> i,
         LKTypeParams<Real> j,
         LJGlobalParams<Real> global) {
        return lk_isotropic_score<Real>::V_dV(
                   dist, bonded_path_length, i, j, global)
            .astuple();
      },
      "dist"_a,
      "bonded_path_length"_a,
      "i"_a,
      "j"_a,
      "global"_a);
}  // namespace potentials

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_potentials<double>(m); }

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
