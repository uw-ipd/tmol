#pragma once

// Compensated summation, for every backend.
//
// Adding many small numbers into one large total in float32 loses the low bits
// of each addend. That costs accuracy, which is the obvious problem, but it
// also makes the total a *rough* function of its inputs: perturb one addend by
// far less than the total's ULP and the result can still jump by several ULP,
// because the error term itself moves. A line search that finite-differences
// the energy reads those jumps as slope.
//
// Neumaier's variant of Kahan summation carries the lost low bits in a second
// accumulator and folds them in once at the end, so the result is very nearly
// the correctly-rounded sum. A perturbation below half an ULP of the total
// then cannot change it at all -- which is the property that matters here, and
// the one a plain float32 accumulation does not have.
//
// Neumaier rather than plain Kahan because the addends here are block-pair
// energies that are frequently larger than the running total early on, which
// is the case plain Kahan gets wrong.
//
// Metal has no float64, so this is how a GPU backend gets a trustworthy total.
// It is also the machinery hbond's degree-10 polynomials would need.

#include "qualifiers.hh"

namespace tmol {
namespace portable {

template <typename T>
struct CompensatedSum {
  T total;
  T correction;

  TMOL_HD CompensatedSum() : total(T(0)), correction(T(0)) {}

  TMOL_HD void add(T x) {
    T const t = total + x;
    // Whichever operand is larger in magnitude keeps its low bits; the
    // other's are what rounding dropped.
    if ((total < T(0) ? -total : total) >= (x < T(0) ? -x : x)) {
      correction += (total - t) + x;
    } else {
      correction += (x - t) + total;
    }
    total = t;
  }

  // Fold the carried low bits in. Call once, at the end.
  TMOL_HD T value() const { return total + correction; }

  // Merge another partial sum, keeping both correction terms.
  TMOL_HD void merge(TMOL_THREAD CompensatedSum<T> const& other) {
    add(other.total);
    correction += other.correction;
  }
};

}  // namespace portable
}  // namespace tmol
