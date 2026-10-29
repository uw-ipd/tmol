#pragma once

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/utility/portable/math.hh>

namespace tmol {
namespace score {
namespace common {

// Number of entries preceding row i of the strict upper triangle of an n-x-n
// matrix.
TMOL_HD int64_t upper_triangle_row_start(int64_t i, int64_t n) {
  return i * (2 * n - i - 1) / 2;
}

TMOL_HD tuple<int, int> upper_triangle_inds_from_linear_index(
    int k,  // the linear index
    int n   // for an n-x-n matrix
) {
  // from
  // https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
  //
  // The closed form needs a square root, and Metal has no double, so the
  // estimate is computed in single precision and then corrected to the exact
  // answer. float carries 24 mantissa bits, so for any n tmol can represent
  // the estimate is off by well under one row and the correction loops run at
  // most one step each -- but they are written as loops so the result is exact
  // regardless of the precision the estimate was computed in.
  int64_t const n64 = n;
  int64_t const discriminant = -8 * int64_t(k) + 4 * n64 * (n64 - 1) - 7;
  int i = n - 2
          - int(tmol::portable::math::floor(
              tmol::portable::math::sqrt(float(discriminant)) / 2.0f - 0.5f));

  if (i < 0) i = 0;
  if (i > n - 2) i = n - 2;
  while (i > 0 && upper_triangle_row_start(i, n64) > int64_t(k)) --i;
  while (i < n - 2 && upper_triangle_row_start(i + 1, n64) <= int64_t(k)) ++i;

  int const j = int(int64_t(k) - upper_triangle_row_start(i, n64) + i + 1);
  return make_tuple(i, j);
}

}  // namespace common
}  // namespace score
}  // namespace tmol
