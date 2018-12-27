#pragma once

#include <tuple>

#include <Eigen/Core>

#include <tmol/score/common/vec.hh>

namespace tmol {
namespace score {
namespace common {

template <int POrd, typename Real>
Real poly_v(const Real& x, const Vec<POrd, Real>& coeffs) {
  static_assert(POrd >= 2);
  Real v = coeffs(0);

#pragma unroll
  for (int i = 1; i < POrd; ++i) {
    v = v * x + coeffs[i];
  }

  return v;
}

template <int POrd, typename Real>
std::tuple<Real, Real> poly_v_d(const Real& x, const Vec<POrd, Real>& coeffs) {
  static_assert(POrd >= 2);
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

}  // namespace common
}  // namespace score
}  // namespace tmol
