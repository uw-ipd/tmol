#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Core>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/tensor/TensorAccessor.h>

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
  weights[0] = Real(1) / Real(6) + fraction * (fraction - Real(1)) / Real(2)
               - weights[3];
  weights[2] = fraction + weights[0] - Real(2) * weights[3];
  weights[1] = Real(1) - weights[0] - weights[2] - weights[3];
}

template <typename Real>
EIGEN_DEVICE_FUNC void cubic_bspline_weights_and_derivatives(
    Real fraction, Real weights[4], Real derivatives[4]) {
  cubic_bspline_weights(fraction, weights);
  derivatives[3] = fraction * fraction / Real(2);
  derivatives[0] = fraction - Real(1) / Real(2) - derivatives[3];
  derivatives[2] = Real(1) + derivatives[0] - fraction * fraction;
  derivatives[1] = -derivatives[0] - derivatives[2] - derivatives[3];
}

// Cubic B-spline stencil of one grid point, shared by every channel sampled
// there. Out-of-bounds indices are clamped; the quadratic penalty (index units)
// supplies the only slope outside, and the spline is constant along such axes.
template <typename Real>
struct GridStencil {
  int index[3][4];
  Real weight[3][4];
  Real weight_derivative[3][4];
  bool outside[3];
  Real penalty;
  Real penalty_gradient[3];
  bool valid;
};

template <typename Real>
EIGEN_DEVICE_FUNC GridStencil<Real> make_grid_stencil(
    Real const position[3], int const dims[3], Real out_of_bounds_penalty) {
  GridStencil<Real> stencil;
  Real const max_index = Real(1 << 30);
  stencil.valid = true;
  for (int axis = 0; axis < 3; ++axis) {
    stencil.valid = stencil.valid && std::isfinite(position[axis])
                    && std::abs(position[axis]) <= max_index;
  }
  stencil.penalty = 0;
  if (!stencil.valid) return stencil;

  Real index[3] = {position[0], position[1], position[2]};
  for (int axis = 0; axis < 3; ++axis) {
    Real offset = 0;
    Real const upper = Real(dims[axis] - 1);
    stencil.outside[axis] = index[axis] < Real(0) || index[axis] > upper;
    if (index[axis] < Real(0)) {
      offset = index[axis];
      index[axis] = Real(0);
    } else if (index[axis] > upper) {
      offset = index[axis] - upper;
      index[axis] = upper;
    }
    stencil.penalty += out_of_bounds_penalty * offset * offset;
    stencil.penalty_gradient[axis] = Real(2) * out_of_bounds_penalty * offset;
    Real const first = std::floor(index[axis] - Real(1));
    cubic_bspline_weights_and_derivatives(
        index[axis] - first - Real(1),
        stencil.weight[axis],
        stencil.weight_derivative[axis]);
    for (int tap = 0; tap < 4; ++tap) {
      stencil.index[axis][tap] =
          mirror_grid_index(static_cast<int>(first) + tap, dims[axis]);
    }
  }
  return stencil;
}

// Value and index-space gradient of one coefficient grid at a stencil.
template <tmol::Device D, typename Real>
EIGEN_DEVICE_FUNC void contract_grid(
    TView<Real, 4, D> const& coefficients,
    int grid,
    GridStencil<Real> const& s,
    Real& value,
    Real gradient[3]) {
  value = 0;
  gradient[0] = gradient[1] = gradient[2] = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        Real const c =
            coefficients[grid][s.index[0][i]][s.index[1][j]][s.index[2][k]];
        value += s.weight[0][i] * s.weight[1][j] * s.weight[2][k] * c;
        gradient[0] +=
            s.weight_derivative[0][i] * s.weight[1][j] * s.weight[2][k] * c;
        gradient[1] +=
            s.weight[0][i] * s.weight_derivative[1][j] * s.weight[2][k] * c;
        gradient[2] +=
            s.weight[0][i] * s.weight[1][j] * s.weight_derivative[2][k] * c;
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (s.outside[axis]) gradient[axis] = 0;
  }
}

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
