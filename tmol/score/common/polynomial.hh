#pragma once

#include <tmol/utility/portable/qualifiers.hh>
#include <tmol/utility/portable/double_float.hh>
#include <tmol/utility/portable/matrix.hh>
#include <tmol/utility/portable/metal_view.hh>

#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

#define def auto TMOL_HD

using tmol::portable::Vec;
// Found by unqualified lookup from the Horner loops below alongside the
// same-precision overload just under this.
using tmol::portable::horner_step;

// One Horner step in ordinary arithmetic, whatever the three precisions are.
//
// Deliberately not `fma`: this is the step the CPU and CUDA builds take, and
// spelling it as a multiply and an add leaves their results bit-identical to
// what they were before the accumulator and coefficient types came apart --
// including the `poly<11, float, double, double>` instantiation, where the
// argument promotes.
//
// The double-float overloads in utility/portable/double_float.hh are more
// specialized, so partial ordering picks them over this one whenever the
// accumulator or the coefficient is a hi/lo pair.
template <typename EvalReal, typename Real, typename CoeffReal>
TMOL_HD EvalReal horner_step(EvalReal v, Real x, CoeffReal c) {
  return v * x + c;
}

// POrd coefficients in descending power order.
//
// Three types, not one. `Real` is the argument and the result, `EvalReal` the
// Horner accumulator, and `CoeffReal` the stored coefficients. All three
// default to Real, so `poly<11, double>` means exactly what it always did and
// the CPU and CUDA instantiations are unchanged.
//
// They come apart for Metal, which has no float64: hbond's degree-10 fits need
// better-than-float32 coefficients or they lose their sign (see
// MPS_PORT_NOTES.md), so that backend instantiates
// `poly<11, float, DoubleFloat<float>, DoubleFloat<float>>` -- float in, float
// out, and the coefficients and accumulator carried as hi + lo pairs.
template <
    int POrd,
    typename Real,
    typename EvalReal = Real,
    typename CoeffReal = Real>
struct poly {
  struct V_dV_T {
    Real V;
    Real dV_dX;

    auto astuple() { return make_tuple(V, dV_dX); }
  };

  static def V(Real X, TMOL_DEVICE_REF(Vec<CoeffReal, POrd> const) coeffs)
      -> Real {
    static_assert(POrd >= 2, "");
    EvalReal v = EvalReal(coeffs(0));

#pragma unroll
    for (int i = 1; i < POrd; ++i) {
      v = horner_step(v, X, coeffs[i]);
    }

    return (Real)v;
  }

  static def V_dV(
      Real X, TMOL_DEVICE_REF(Vec<CoeffReal, POrd> const) coeffs)
      -> V_dV_T {
    static_assert(POrd >= 2, "");
    EvalReal v = EvalReal(coeffs(0));
    EvalReal d = EvalReal(coeffs(0));

#pragma unroll
    for (int i = 1; i < POrd - 1; ++i) {
      v = horner_step(v, X, coeffs[i]);
      d = horner_step(d, X, v);
    }

    v = horner_step(v, X, coeffs[POrd - 1]);

    return {(Real)v, (Real)d};
  }
};

template <
    int PDim,
    typename Real,
    typename EvalReal = Real,
    typename CoeffReal = Real>
struct bound_poly {
  typedef typename poly<PDim, Real, EvalReal, CoeffReal>::V_dV_T V_dV_T;

  static def V(
      TMOL_THREAD const Real& x,
      TMOL_DEVICE_REF(Vec<CoeffReal, PDim> const) coeffs,
      TMOL_DEVICE_REF(Vec<CoeffReal, 2> const) range,
      TMOL_DEVICE_REF(Vec<CoeffReal, 2> const) bound) -> Real {
    // range and bound are stored beside the coefficients, so they share their
    // type; the comparison and the result are in the working precision.
    if (x < Real(range[0])) {
      return Real(bound[0]);
    } else if (x > Real(range[1])) {
      return Real(bound[1]);
    } else {
      return poly<PDim, Real, EvalReal, CoeffReal>::V(x, coeffs);
    }
  }

  static def V_dV(
      Real x,
      TMOL_DEVICE_REF(Vec<CoeffReal, PDim> const) coeffs,
      TMOL_DEVICE_REF(Vec<CoeffReal, 2> const) range,
      TMOL_DEVICE_REF(Vec<CoeffReal, 2> const) bound) -> V_dV_T {
    if (x < Real(range[0])) {
      return {Real(bound[0]), 0};
    } else if (x > Real(range[1])) {
      return {Real(bound[1]), 0};
    } else {
      return poly<PDim, Real, EvalReal, CoeffReal>::V_dV(x, coeffs);
    }
  }
};

#undef def

}  // namespace common
}  // namespace score
}  // namespace tmol
