#pragma once

#include <Eigen/Core>

#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <int POrd, typename Real>
Real poly_v(Real x, Vec<Real, POrd> coeffs) {
  static_assert(POrd >= 2, "");
  Real v = coeffs(0);

#pragma unroll
  for (int i = 1; i < POrd; ++i) {
    v = v * x + coeffs[i];
  }

  return v;
}

template <int POrd, typename Real>
auto poly_v_d(Real x, Vec<Real, POrd> coeffs) -> tuple<Real, Real> {
  static_assert(POrd >= 2, "");
  Real v = coeffs(0);
  Real d = coeffs(0);

#pragma unroll
  for (int i = 1; i < POrd - 1; ++i) {
    v = v * x + coeffs[i];
    d = d * x + v;
  }

  v = v * x + coeffs[POrd - 1];

  return {v, d};
}

template <int PDim, typename Real>
Real bound_poly_V(
    const Real& x,
    const Vec<Real, PDim>& coeffs,
    const Vec<Real, 2>& range,
    const Vec<Real, 2>& bound) {
  if (x < range[0]) {
    return bound[0];
  } else if (x > range[1]) {
    return bound[1];
  } else {
    return poly_v(x, coeffs);
  }
}

template <int PDim, typename Real>
auto bound_poly_V_dV(
    Real x, Vec<Real, PDim> coeffs, Vec<Real, 2> range, Vec<Real, 2> bound)
    -> tuple<Real, Real> {
  if (x < range[0]) {
    return {bound[0], 0};
  } else if (x > range[1]) {
    return {bound[1], 0};
  } else {
    return poly_v_d(x, coeffs);
  }
}

}  // namespace common
}  // namespace score
}  // namespace tmol
