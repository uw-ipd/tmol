#pragma once

// An unevaluated sum of two floats, for every backend.
//
// hbond fits degree-10 polynomials whose coefficients run to 4.6e4 and cancel
// heavily. Rounding those coefficients to float32 costs a *relative* error of
// 1.96 -- 200% -- at points where the polynomial nearly vanishes, which is why
// the term evaluates them in float64 and why it stayed on the host when Metal,
// which has no float64 at all, was ported.
//
// Measured on all 58 polynomials from the database: the coefficients are what
// need the precision, not the argument. Float64 x with float32 coefficients is
// still 1.96; float32 x with float64 coefficients is 2.2e-03. So carrying the
// coefficients and the Horner accumulator as hi + lo float32 pairs recovers
// float64-coefficient accuracy exactly -- it lands on the same 2.2e-03, the
// floor set by having a float32 argument at all.
//
// Not device-specific: on CPU and CUDA this is as valid as it is on Metal, and
// score/common/polynomial.hh already separates the accumulator type from the
// coefficient type, so it drops into that seam.
//
// The error-free transformations below assume the compiler does not reassociate
// float arithmetic. Metal compiles with MTLMathModeSafe by default, so that
// holds; the two-product uses fma rather than Dekker splitting so it stays
// exact even where contraction is permitted.

#include "math.hh"
#include "qualifiers.hh"

namespace tmol {
namespace portable {

template <typename Real>
struct DoubleFloat {
  Real hi;
  Real lo;

  TMOL_HD DoubleFloat() : hi(0), lo(0) {}
  TMOL_HD DoubleFloat(Real value) : hi(value), lo(0) {}
  TMOL_HD DoubleFloat(Real high, Real low) : hi(high), lo(low) {}

  // The value, rounded once to the working precision.
  TMOL_HD Real value() const { return hi + lo; }

  // Explicit, so a double-float never collapses to one float by accident.
  // score/common/polynomial.hh casts its accumulator on the way out.
  TMOL_HD explicit operator Real() const { return hi + lo; }
};

// Knuth's two-sum: s is the rounded sum and e the exact rounding error, so
// a + b == s + e with no loss. Branch-free, and valid for any magnitudes.
template <typename Real>
TMOL_HD void two_sum(Real a, Real b, TMOL_THREAD Real& s, TMOL_THREAD Real& e) {
  s = a + b;
  Real const bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// Dekker's two-product via fma: p is the rounded product and e the exact
// rounding error.
template <typename Real>
TMOL_HD void two_prod(
    Real a, Real b, TMOL_THREAD Real& p, TMOL_THREAD Real& e) {
  p = a * b;
  e = tmol::portable::math::fma(a, b, -p);
}

template <typename Real>
TMOL_HD DoubleFloat<Real> renormalize(Real hi, Real lo) {
  Real s, e;
  two_sum(hi, lo, s, e);
  return DoubleFloat<Real>(s, e);
}

// The three operations a Horner recurrence needs: double-float times plain,
// double-float plus plain, and double-float plus double-float.
template <typename Real>
TMOL_HD DoubleFloat<Real> operator*(DoubleFloat<Real> a, Real b) {
  Real p, e;
  two_prod(a.hi, b, p, e);
  return renormalize(p, e + a.lo * b);
}

template <typename Real>
TMOL_HD DoubleFloat<Real> operator+(DoubleFloat<Real> a, Real b) {
  Real s, e;
  two_sum(a.hi, b, s, e);
  return renormalize(s, e + a.lo);
}

template <typename Real>
TMOL_HD DoubleFloat<Real> operator+(DoubleFloat<Real> a, DoubleFloat<Real> b) {
  Real s, e;
  two_sum(a.hi, b.hi, s, e);
  return renormalize(s, e + (a.lo + b.lo));
}

// One Horner step, v <- v * x + c, with the coefficient carried as a
// double-float.
//
// Spelled as a named operation rather than built out of the operators above
// because the operators renormalize on every result, and a Horner step that
// multiplies and then adds renormalizes twice in a row for no gain. Fusing
// them takes the step from about twenty-four float operations to eighteen.
// score/common/polynomial.hh calls this; the plain-precision overload lives
// there, so the Horner loop is written once for every combination of
// accumulator and coefficient precision.
template <typename Real>
TMOL_HD DoubleFloat<Real> horner_step(
    DoubleFloat<Real> v, Real x, DoubleFloat<Real> c) {
  Real p, e;
  two_prod(v.hi, x, p, e);
  Real lo = e + v.lo * x;
  Real s, e2;
  two_sum(p, c.hi, s, e2);
  return renormalize(s, (lo + e2) + c.lo);
}

// The same step with a plain accumulator and a double-float coefficient.
//
// The coefficient's two halves are added separately rather than pre-rounded to
// one float: the low half survives whenever |v * x| is comparable to |c.hi|,
// which is where a degree-10 fit cancels. This is the cheap configuration --
// two operations a step against eighteen -- and whether it is accurate enough
// is a question about a particular polynomial family, not a general one, so
// the caller chooses; hbond measured it and chose the accurate one.
template <typename Real>
TMOL_HD Real horner_step(Real v, Real x, DoubleFloat<Real> c) {
  return tmol::portable::math::fma(v, x, c.hi) + c.lo;
}

}  // namespace portable
}  // namespace tmol
