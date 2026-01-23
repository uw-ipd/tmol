#pragma once

#include <Eigen/Core>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

#define def                \
  template <typename Real> \
  auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

def interpolate_t(Real t, Real p0, Real dp0, Real p1, Real dp1)->Real {
  // Cubic interpolation of p on t in [0, 1].

  // clang-format off
  return p0 + t * (
      dp0 + t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 + t * (dp0 + dp1 + 2 * p0 - 2 * p1))
  );
  // clang-format on
}

def interpolate_dt(Real t, Real p0, Real dp0, Real p1, Real dp1)->Real {
  // Cubic interpolation of dp/dt on t in [0, 1].

  // clang-format off
  return dp0 + t * (
      -4 * dp0 - 2 * dp1 - 6 * p0 + 6 * p1 + t * (3 * dp0 + 3 * dp1 + 6 * p0 - 6 * p1)
  );
  // clang-format on
}

def interpolate(
    Real x, Real x0, Real p0, Real dpdx0, Real x1, Real p1, Real dpdx1)
    ->Real {
  // Cubic interpolation of p on x in [x0, x1].
  Real t = (x - x0) / (x1 - x0);
  Real dp0 = dpdx0 * (x1 - x0);
  Real dp1 = dpdx1 * (x1 - x0);

  // clang-format off
  return p0 + t * (
      dp0 + t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 + t * (dp0 + dp1 + 2 * p0 - 2 * p1))
  );
  // clang-format on
}

def interpolate_dx(
    Real x, Real x0, Real p0, Real dpdx0, Real x1, Real p1, Real dpdx1)
    ->Real {
  // Cubic interpolation of dp/dx on x in [x0, x1].
  Real t = (x - x0) / (x1 - x0);
  Real dp0 = dpdx0 * (x1 - x0);
  Real dp1 = dpdx1 * (x1 - x0);

  // clang-format off
  Real dp = dp0 + t * (
      -4 * dp0 - 2 * dp1 - 6 * p0 + 6 * p1 + t * (3 * dp0 + 3 * dp1 + 6 * p0 - 6 * p1)
  );
  // clang-format on

  return dp / (x1 - x0);
}

def interpolate_V_dV(
    Real x, Real x0, Real p0, Real dpdx0, Real x1, Real p1, Real dpdx1)
    ->tuple<Real, Real> {
  // Cubic interpolation of p, dp/dx on x in [x0, x1].
  Real t = (x - x0) / (x1 - x0);
  Real dp0 = dpdx0 * (x1 - x0);
  Real dp1 = dpdx1 * (x1 - x0);

  // clang-format off
  Real p = p0 + t * (
      dp0 + t * (-2 * dp0 - dp1 - 3 * p0 + 3 * p1 + t * (dp0 + dp1 + 2 * p0 - 2 * p1))
  );
  Real dp = dp0 + t * (
      -4 * dp0 - 2 * dp1 - 6 * p0 + 6 * p1 + t * (3 * dp0 + 3 * dp1 + 6 * p0 - 6 * p1)
  );
  // clang-format on

  return {p, dp / (x1 - x0)};
}

def interpolate_to_zero_t(Real t, Real p0, Real dp0)->Real {
  // Cubic interpolation of p on t in [0, 1] to (p1, dp1) == 0.

  // clang-format off
  return p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0)));
  // clang-format on
}

def interpolate_to_zero_dt(Real t, Real p0, Real dp0)->Real {
  // Cubic interpolation of dp/dt on t in [0, 1] to (p1, dp1) == 0.

  // clang-format off
  return dp0 + t * (-4 * dp0 - 6 * p0 + t * (3 * dp0 + 6 * p0));
  // clang-format on
}

def interpolate_to_zero(Real x, Real x0, Real p0, Real dpdx0, Real x1)->Real {
  // Cubic interpolation of p on x in [x0, x1] to (p1, dpdx1) == 0 at x1
  Real t = (x - x0) / (x1 - x0);
  Real dp0 = dpdx0 * (x1 - x0);

  // clang-format off
  return p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0)));
  // clang-format on
}

def interpolate_to_zero_dx(Real x, Real x0, Real p0, Real dpdx0, Real x1)
    ->Real {
  // Cubic interpolation of dp/dx on x in [x0, x1] to (p1, dpdx1) == 0 at x1.
  Real t = (x - x0) / (x1 - x0);
  Real dp0 = dpdx0 * (x1 - x0);

  // clang-format off
  Real dp = dp0 + t * (-4 * dp0 - 6 * p0 + t * (3 * dp0 + 6 * p0));
  // clang-format on

  return dp / (x1 - x0);
}

def interpolate_to_zero_V_dV(Real x, Real x0, Real p0, Real dpdx0, Real x1)
    ->tuple<Real, Real> {
  // Cubic interpolation of dp/dx on x in [x0, x1] to (p1, dpdx1) == 0 at x1.
  Real t = (x - x0) / (x1 - x0);
  Real dp0 = dpdx0 * (x1 - x0);

  // clang-format off
  return {
    p0 + t * (dp0 + t * (-2 * dp0 - 3 * p0 + t * (dp0 + 2 * p0))),
    (dp0 + t * (-4 * dp0 - 6 * p0 + t * (3 * dp0 + 6 * p0))) / (x1 - x0)
  };
  // clang-format on
}

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
