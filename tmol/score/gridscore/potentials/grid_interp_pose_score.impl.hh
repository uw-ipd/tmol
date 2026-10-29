#pragma once

#include <cmath>
#include <limits>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/grid_interp_point.hh>
#include <tmol/score/gridscore/potentials/grid_interp_pose_score.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GridInterpDispatch<DeviceOps, D, Real, Int>::forward_with_derivatives(
    ContextManager& mgr,
    TView<Real, 4, D> coefficients,
    TView<Vec<Real, 3>, 1, D> indices,
    TView<Int, 1, D> grid_indices,
    Real out_of_bounds_penalty)
    -> std::tuple<
        TPack<Real, 1, D>,
        TPack<Real, 2, D>,
        TPack<Real, 1, D>,
        TPack<Real, 2, D>> {
  int const n_points = indices.size(0);
  Int const nx = coefficients.size(1);
  Int const ny = coefficients.size(2);
  Int const nz = coefficients.size(3);
  Int const n_grids = coefficients.size(0);
  auto values_t = TPack<Real, 1, D>::empty({n_points});
  auto derivatives_t = TPack<Real, 2, D>::empty({n_points, 3});
  auto penalties_t = TPack<Real, 1, D>::empty({n_points});
  auto penalty_derivatives_t = TPack<Real, 2, D>::empty({n_points, 3});
  auto values = values_t.view;
  auto derivatives = derivatives_t.view;
  auto penalties = penalties_t.view;
  auto penalty_derivatives = penalty_derivatives_t.view;

  LAUNCH_BOX_32;
  auto interpolate_point = [=] TMOL_DEVICE_FUNC(int point) {
    int const int_dims[3] = {int(nx), int(ny), int(nz)};
    Real const position[3] = {
        indices[point][0], indices[point][1], indices[point][2]};
    Int const grid = grid_indices[point];
    auto const stencil =
        make_grid_stencil<Real>(position, int_dims, out_of_bounds_penalty);
    if (grid < 0 || grid >= n_grids || !stencil.valid) {
      Real const nan = std::numeric_limits<Real>::quiet_NaN();
      values[point] = nan;
      penalties[point] = nan;
      for (int axis = 0; axis < 3; ++axis) {
        derivatives[point][axis] = nan;
        penalty_derivatives[point][axis] = nan;
      }
      return;
    }
    Real value, gradient[3];
    contract_grid<D, Real>(coefficients, grid, stencil, value, gradient);
    values[point] = value;
    penalties[point] = stencil.penalty;
    for (int axis = 0; axis < 3; ++axis) {
      derivatives[point][axis] = gradient[axis];
      penalty_derivatives[point][axis] = stencil.penalty_gradient[axis];
    }
  };
  DeviceOps<D>::template forall<launch_t>(mgr, n_points, interpolate_point);
  return std::make_tuple(
      values_t, derivatives_t, penalties_t, penalty_derivatives_t);
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
