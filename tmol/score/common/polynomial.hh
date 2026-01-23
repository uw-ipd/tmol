#pragma once

#include <Eigen/Core>

#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <int POrd, typename Real, typename EvalReal = Real>
struct poly {
  struct V_dV_T {
    Real V;
    Real dV_dX;

    auto astuple() { return make_tuple(V, dV_dX); }
  };

  static def V(Real X, Vec<Real, POrd> const& coeffs)->Real {
    static_assert(POrd >= 2, "");
    EvalReal v = coeffs(0);

#pragma unroll
    for (int i = 1; i < POrd; ++i) {
      v = v * X + coeffs[i];
    }

    return (Real)v;
  }

  static def V_dV(Real X, Vec<Real, POrd> const& coeffs)->V_dV_T {
    static_assert(POrd >= 2, "");
    EvalReal v = coeffs(0);
    EvalReal d = coeffs(0);

#pragma unroll
    for (int i = 1; i < POrd - 1; ++i) {
      v = v * X + coeffs[i];
      d = d * X + v;
    }

    v = v * X + coeffs[POrd - 1];

    return {(Real)v, (Real)d};
  }
};

template <int PDim, typename Real, typename EvalReal = Real>
struct bound_poly {
  typedef typename poly<PDim, Real, EvalReal>::V_dV_T V_dV_T;

  static def V(
      const Real& x,
      const Vec<Real, PDim>& coeffs,
      const Vec<Real, 2>& range,
      const Vec<Real, 2>& bound)
      ->Real {
    if (x < range[0]) {
      return bound[0];
    } else if (x > range[1]) {
      return bound[1];
    } else {
      return poly<PDim, Real, EvalReal>::V(x, coeffs);
    }
  }

  static def V_dV(
      Real x,
      Vec<Real, PDim> const& coeffs,
      Vec<Real, 2> const& range,
      Vec<Real, 2> const& bound)
      ->V_dV_T {
    if (x < range[0]) {
      return {bound[0], 0};
    } else if (x > range[1]) {
      return {bound[1], 0};
    } else {
      return poly<PDim, Real, EvalReal>::V_dV(x, coeffs);
    }
  }
};

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
