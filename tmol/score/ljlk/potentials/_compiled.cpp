#include <pybind11/eigen.h>
#include <torch/torch.h>

#include "lj.hh"
#include "lk_isotropic.hh"

using namespace tmol::score::ljlk::potentials;

// clang-format off
//
#define pyarg(v) py::arg(#v)

#define GLOBAL_PARAMS()      \
  Real lj_hbond_dis,         \
  Real lj_hbond_OH_donor_dis,\
  Real lj_hbond_hdis         \

#define GLOBAL_PARAM_STRUCT() \
{                        \
  lj_hbond_dis,          \
  lj_hbond_OH_donor_dis, \
  lj_hbond_hdis          \
}

#define GLOBAL_PYARGS() \
  pyarg(lj_hbond_dis),          \
  pyarg(lj_hbond_OH_donor_dis), \
  pyarg(lj_hbond_hdis)

#define LJ_PARAMS(N)   \
  Real N##_lj_radius,   \
  Real N##_lj_wdepth,   \
  bool N##_is_donor,    \
  bool N##_is_hydroxyl, \
  bool N##_is_polarh,   \
  bool N##_is_acceptor

#define LJ_PARAM_STRUCT(N)   \
{               \
  N##_lj_radius,   \
  N##_lj_wdepth,   \
  N##_is_donor,    \
  N##_is_hydroxyl, \
  N##_is_polarh,   \
  N##_is_acceptor  \
}

#define LJ_PYARGS(N)       \
  pyarg(N##_lj_radius),  \
  pyarg(N##_lj_wdepth),  \
  pyarg(N##_is_donor),   \
  pyarg(N##_is_hydroxyl),\
  pyarg(N##_is_polarh),  \
  pyarg(N##_is_acceptor)

#define LK_PARAMS(N)   \
  Real N##_lj_radius,   \
  Real N##_lk_dgfree,   \
  Real N##_lk_lambda,   \
  Real N##_lk_volume,   \
  bool N##_is_donor,    \
  bool N##_is_hydroxyl, \
  bool N##_is_polarh,   \
  bool N##_is_acceptor

#define LK_PARAM_STRUCT(N)   \
{                  \
  N##_lj_radius,   \
  N##_lk_dgfree,   \
  N##_lk_lambda,   \
  N##_lk_volume,   \
  N##_is_donor,    \
  N##_is_hydroxyl, \
  N##_is_polarh,   \
  N##_is_acceptor  \
}

#define LK_PYARGS(N)       \
  pyarg(N##_lj_radius),  \
  pyarg(N##_lk_dgfree),  \
  pyarg(N##_lk_lambda),  \
  pyarg(N##_lk_volume),  \
  pyarg(N##_is_donor),   \
  pyarg(N##_is_hydroxyl),\
  pyarg(N##_is_polarh),  \
  pyarg(N##_is_acceptor)
// clang-format on

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "lj_sigma",
      py::vectorize([](LJ_PARAMS(i), LJ_PARAMS(j), GLOBAL_PARAMS()) {
        return lj_sigma<Real, LJTypeParams<Real>, LJGlobalParams<Real>>(
            LJ_PARAM_STRUCT(i), LJ_PARAM_STRUCT(j), GLOBAL_PARAM_STRUCT());
      }),

      LJ_PYARGS(i),
      LJ_PYARGS(j),
      GLOBAL_PYARGS());

  m.def("vdw_V_dV", &vdw_V_dV<Real>, "dist"_a, "sigma"_a, "epsilon"_a);
  m.def("vdw_V", &vdw_V<Real>, "dist"_a, "sigma"_a, "epsilon"_a);

  m.def(
      "lj_score_V",
      [](Real dist,
         Real bonded_path_length,
         LJ_PARAMS(i),
         LJ_PARAMS(j),
         GLOBAL_PARAMS()) {
        return lj_score_V(
            dist,
            bonded_path_length,
            LJ_PARAM_STRUCT(i),
            LJ_PARAM_STRUCT(j),
            GLOBAL_PARAM_STRUCT());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LJ_PYARGS(i),
      LJ_PYARGS(j),
      GLOBAL_PYARGS());

  m.def(
      "lj_score_V_dV",
      [](Real dist,
         Real bonded_path_length,
         LJ_PARAMS(i),
         LJ_PARAMS(j),
         GLOBAL_PARAMS()) {
        return lj_score_V_dV(
            dist,
            bonded_path_length,
            LJ_PARAM_STRUCT(i),
            LJ_PARAM_STRUCT(j),
            GLOBAL_PARAM_STRUCT());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LJ_PYARGS(i),
      LJ_PYARGS(j),
      GLOBAL_PYARGS());

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
         LK_PARAMS(i),
         LK_PARAMS(j),
         GLOBAL_PARAMS()) {
        return lk_isotropic_score_V(
            dist,
            bonded_path_length,
            LK_PARAM_STRUCT(i),
            LK_PARAM_STRUCT(j),
            GLOBAL_PARAM_STRUCT());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LK_PYARGS(i),
      LK_PYARGS(j),
      GLOBAL_PYARGS());

  m.def(
      "lk_isotropic_score_V_dV",
      [](Real dist,
         Real bonded_path_length,
         LK_PARAMS(i),
         LK_PARAMS(j),
         GLOBAL_PARAMS()) {
        return lk_isotropic_score_V_dV(
            dist,
            bonded_path_length,
            LK_PARAM_STRUCT(i),
            LK_PARAM_STRUCT(j),
            GLOBAL_PARAM_STRUCT());
      },
      "dist"_a,
      "bonded_path_length"_a,
      LK_PYARGS(i),
      LK_PYARGS(j),
      GLOBAL_PYARGS());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_potentials<double>(m);
}
