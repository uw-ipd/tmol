#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/extension.h>

#include <tmol/score/elec/potentials/potentials.hh>
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

  m.def(
      "elec_V",
      [](Real dist,
         Real charge_i,
         Real charge_j,
         Real bonded_path_length,
         float D,
         float D0,
         float S,
         float min_dis,
         float max_dis) {
        using tmol::score::elec::potentials::deps_ddist;
        using tmol::score::elec::potentials::ElecGlobalParams;
        using tmol::score::elec::potentials::eps;

        Real cutoff_offset =
            Real(322.0637) / (max_dis * eps<Real>(max_dis, D, D0, S));
        Real low_end = min_dis + Real(0.25);
        Real low_eps = eps<Real>(low_end, D, D0, S);
        Real high_start = max_dis - Real(1);
        Real high_eps = eps<Real>(high_start, D, D0, S);
        ElecGlobalParams<Real> params{
            D,
            D0,
            S,
            min_dis,
            max_dis,
            cutoff_offset,
            Real(322.0637) / (min_dis * eps<Real>(min_dis, D, D0, S))
                - cutoff_offset,
            Real(322.0637) / (low_end * low_eps) - cutoff_offset,
            -Real(322.0637)
                * (low_eps + low_end * deps_ddist<Real>(low_end, D, D0, S))
                / (low_end * low_end * low_eps * low_eps),
            Real(322.0637) / (high_start * high_eps) - cutoff_offset,
            -Real(322.0637)
                * (high_eps
                   + high_start * deps_ddist<Real>(high_start, D, D0, S))
                / (high_start * high_start * high_eps * high_eps)};
        return tmol::score::elec::potentials::elec<Real>(
            dist, charge_i, charge_j, bonded_path_length, params);
      },
      "dist"_a,
      "charge_i"_a,
      "charge_j"_a,
      "bonded_path_length"_a,
      "D"_a,
      "D0"_a,
      "S"_a,
      "min_dis"_a,
      "max_dis"_a);
}  // namespace potentials

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_potentials<double>(m); }

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
