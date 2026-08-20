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

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GridInterpDispatch<DeviceOps, D, Real, Int>::forward(
    ContextManager& mgr,
    TView<Real, 4, D> coefficients,
    TView<Vec<Real, 3>, 1, D> indices,
    TView<Int, 1, D> grid_indices) -> TPack<Real, 1, D> {
  int const n_points = indices.size(0);
  Int const nx = coefficients.size(1);
  Int const ny = coefficients.size(2);
  Int const nz = coefficients.size(3);
  Int const n_grids = coefficients.size(0);
  auto output_t = TPack<Real, 1, D>::empty({n_points});
  auto output = output_t.view;

  LAUNCH_BOX_32;
  auto interpolate_point = [=] TMOL_DEVICE_FUNC(int point) {
    Vec<Real, 3> const index = indices[point];
    Int const grid = grid_indices[point];
    // Guard the double->int64 cast below: reject non-finite AND finite-but-huge
    // indices whose floor would overflow Int (undefined behavior). The bound
    // (2^30) is far beyond any real grid dimension, so valid ligand coordinates
    // are never rejected; pathological inputs yield NaN instead of UB.
    Real const max_index = Real(1 << 30);
    if (grid < 0 || grid >= n_grids
        || !std::isfinite(index[0]) || std::abs(index[0]) > max_index
        || !std::isfinite(index[1]) || std::abs(index[1]) > max_index
        || !std::isfinite(index[2]) || std::abs(index[2]) > max_index) {
      output[point] = std::numeric_limits<Real>::quiet_NaN();
      return;
    }

    Int first[3];
    Real weights[3][4];
    for (int axis = 0; axis < 3; ++axis) {
      Real const first_real = std::floor(index[axis] - Real(1));
      first[axis] = static_cast<Int>(first_real);
      cubic_bspline_weights(index[axis] - first_real - Real(1), weights[axis]);
    }

    Real value = 0;
    for (int x_offset = 0; x_offset < 4; ++x_offset) {
      Int const x = mirror_grid_index(first[0] + x_offset, nx);
      for (int y_offset = 0; y_offset < 4; ++y_offset) {
        Int const y = mirror_grid_index(first[1] + y_offset, ny);
        Real const xy_weight = weights[0][x_offset] * weights[1][y_offset];
        for (int z_offset = 0; z_offset < 4; ++z_offset) {
          Int const z = mirror_grid_index(first[2] + z_offset, nz);
          value += xy_weight * weights[2][z_offset]
                   * coefficients[grid][x][y][z];
        }
      }
    }
    output[point] = value;
  };
  DeviceOps<D>::template forall<launch_t>(mgr, n_points, interpolate_point);
  return output_t;
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
