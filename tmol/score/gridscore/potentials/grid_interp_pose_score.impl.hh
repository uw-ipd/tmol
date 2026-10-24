#pragma once

#include <cmath>
#include <limits>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/gridscore/potentials/grid_interp_pose_score.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Int>
EIGEN_DEVICE_FUNC Int mirror_grid_index(Int index, Int size) {
  if (size <= 1) return 0;
  Int const period = 2 * size - 2;
  index %= period;
  if (index < 0) index += period;
  return index < size ? index : period - index;
}

template <typename Real>
EIGEN_DEVICE_FUNC void cubic_bspline_weights(Real fraction, Real weights[4]) {
  weights[3] = fraction * fraction * fraction / Real(6);
  weights[0] = Real(1) / Real(6)
               + fraction * (fraction - Real(1)) / Real(2) - weights[3];
  weights[2] = fraction + weights[0] - Real(2) * weights[3];
  weights[1] = Real(1) - weights[0] - weights[2] - weights[3];
}

template <typename Real>
EIGEN_DEVICE_FUNC void cubic_bspline_weights_and_derivatives(
    Real fraction, Real weights[4], Real derivatives[4]) {
  cubic_bspline_weights(fraction, weights);
  derivatives[3] = fraction * fraction / Real(2);
  derivatives[0] =
      fraction - Real(1) / Real(2) - derivatives[3];
  derivatives[2] =
      Real(1) + derivatives[0] - fraction * fraction;
  derivatives[1] =
      -derivatives[0] - derivatives[2] - derivatives[3];
}

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
    Int const dims[3] = {nx, ny, nz};
    Vec<Real, 3> index = indices[point];
    Int const grid = grid_indices[point];
    Real const max_index = Real(1 << 30);
    if (grid < 0 || grid >= n_grids
        || !std::isfinite(index[0]) || std::abs(index[0]) > max_index
        || !std::isfinite(index[1]) || std::abs(index[1]) > max_index
        || !std::isfinite(index[2]) || std::abs(index[2]) > max_index) {
      Real const nan = std::numeric_limits<Real>::quiet_NaN();
      values[point] = nan;
      penalties[point] = nan;
      for (int axis = 0; axis < 3; ++axis) {
        derivatives[point][axis] = nan;
        penalty_derivatives[point][axis] = nan;
      }
      return;
    }

    bool outside[3] = {false, false, false};
    Real penalty = 0;
    for (int axis = 0; axis < 3; ++axis) {
      Real offset = 0;
      Real const upper = Real(dims[axis] - 1);
      if (index[axis] < Real(0)) {
        offset = index[axis];
        index[axis] = Real(0);
        outside[axis] = true;
      } else if (index[axis] > upper) {
        offset = index[axis] - upper;
        index[axis] = upper;
        outside[axis] = true;
      }
      penalty += out_of_bounds_penalty * offset * offset;
      penalty_derivatives[point][axis] =
          Real(2) * out_of_bounds_penalty * offset;
    }

    Int first[3];
    Real weights[3][4];
    Real weight_derivatives[3][4];
    for (int axis = 0; axis < 3; ++axis) {
      Real const first_real = std::floor(index[axis] - Real(1));
      first[axis] = static_cast<Int>(first_real);
      cubic_bspline_weights_and_derivatives(
          index[axis] - first_real - Real(1),
          weights[axis],
          weight_derivatives[axis]);
    }

    Int stencil[3][4];
    for (int axis = 0; axis < 3; ++axis) {
      for (int tap = 0; tap < 4; ++tap) {
        stencil[axis][tap] = mirror_grid_index(first[axis] + tap, dims[axis]);
      }
    }

    Real value = 0;
    Real gradient[3] = {Real(0), Real(0), Real(0)};
    for (int x_offset = 0; x_offset < 4; ++x_offset) {
      for (int y_offset = 0; y_offset < 4; ++y_offset) {
        for (int z_offset = 0; z_offset < 4; ++z_offset) {
          Real const coefficient =
              coefficients[grid][stencil[0][x_offset]][stencil[1][y_offset]]
                          [stencil[2][z_offset]];
          value += weights[0][x_offset] * weights[1][y_offset]
                   * weights[2][z_offset] * coefficient;
          gradient[0] += weight_derivatives[0][x_offset]
                         * weights[1][y_offset] * weights[2][z_offset]
                         * coefficient;
          gradient[1] += weights[0][x_offset]
                         * weight_derivatives[1][y_offset]
                         * weights[2][z_offset] * coefficient;
          gradient[2] += weights[0][x_offset] * weights[1][y_offset]
                         * weight_derivatives[2][z_offset] * coefficient;
        }
      }
    }
    values[point] = value;
    penalties[point] = penalty;
    for (int axis = 0; axis < 3; ++axis) {
      // move_to_boundary makes the clamped spline constant along an axis once
      // it is outside. The quadratic penalty supplies the only OOB slope.
      derivatives[point][axis] = outside[axis] ? Real(0) : gradient[axis];
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
