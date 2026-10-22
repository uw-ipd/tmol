#pragma once

#include <Eigen/Core>

#include <tmol/numeric/bspline_compiled/bspline.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/density/potentials/density_score.hh>

namespace tmol {
namespace score {
namespace density {
namespace potentials {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
auto DensityScoreDispatch<DeviceDispatch, D, Real>::forward(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> coords,
    TView<Real, 3, D> coeffs,
    TView<Real, 1, D> origin,
    TView<Real, 2, D> inv_basis,
    TView<Real, 1, D> pad,
    bool periodic,
    TView<int64_t, 1, D> atom_index,
    TView<int64_t, 1, D> atom_group,
    TView<Real, 1, D> atom_weight,
    int64_t n_groups)
    -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>> {
  using tmol::score::common::accumulate;
  typedef Vec<Real, 3> Real3;
  typedef tmol::numeric::bspline::ndspline<3, 3, D, Real, int32_t> spline;

  int const n_atoms = atom_index.size(0);
  auto energy_t = TPack<Real, 1, D>::zeros({n_groups});
  auto dE_dx_t = TPack<Real3, 1, D>::zeros({n_atoms});
  auto energy = energy_t.view;
  auto dE_dx = dE_dx_t.view;

  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_atoms, [=] TMOL_DEVICE_FUNC(int a) {
        Real3 const delta =
            coords[atom_index[a]] - Real3(origin[0], origin[1], origin[2]);
        Real3 u;
        for (int i = 0; i < 3; ++i) {
          u[i] = pad[i] + inv_basis[i][0] * delta[0]
                 + inv_basis[i][1] * delta[1] + inv_basis[i][2] * delta[2];
        }
        if (!periodic) {
          // the open map is zero beyond the padded grid's spline support
          for (int i = 0; i < 3; ++i) {
            if (!(u[i] >= 1 && u[i] <= coeffs.size(i) - 3)) return;
          }
        }
        Real value;
        Real3 dvalue_du;
        tmol::score::common::tie(value, dvalue_du) =
            spline::interpolate(coeffs, u);
        Real const w = atom_weight[a];
        accumulate<D, Real>::add(energy[atom_group[a]], -w * value);
        Real3 grad;
        for (int j = 0; j < 3; ++j) {
          grad[j] =
              -w
              * (inv_basis[0][j] * dvalue_du[0] + inv_basis[1][j] * dvalue_du[1]
                 + inv_basis[2][j] * dvalue_du[2]);
        }
        dE_dx[a] = grad;
      });
  return {energy_t, dE_dx_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real>
auto DensityScoreDispatch<DeviceDispatch, D, Real>::backward(
    ContextManager& mgr,
    TView<Vec<Real, 3>, 1, D> dE_dx,
    TView<int64_t, 1, D> atom_index,
    TView<int64_t, 1, D> atom_group,
    TView<Real, 1, D> group_grad,
    int64_t n_coords) -> TPack<Vec<Real, 3>, 1, D> {
  int const n_atoms = atom_index.size(0);
  auto grad_t = TPack<Vec<Real, 3>, 1, D>::zeros({n_coords});
  auto grad = grad_t.view;

  LAUNCH_BOX_128;
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_atoms, [=] TMOL_DEVICE_FUNC(int a) {
        grad[atom_index[a]] = group_grad[atom_group[a]] * dE_dx[a];
      });
  return grad_t;
}

}  // namespace potentials
}  // namespace density
}  // namespace score
}  // namespace tmol
