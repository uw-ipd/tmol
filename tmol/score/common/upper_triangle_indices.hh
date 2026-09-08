#pragma once

#include <cstdint>

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/tuple.hh>

namespace tmol {
namespace score {
namespace common {

inline tuple<int, int> TMOL_DEVICE_FUNC upper_triangle_inds_from_linear_index(
    int k,  // the linear index
    int n   // for an n-x-n matrix
) {
  // from
  // https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
  int64_t const discriminant =
      -8 * int64_t(k) + 4 * int64_t(n) * int64_t(n - 1) - 7;
  int const i = n - 2 - int(floor(sqrt(double(discriminant)) / 2.0 - 0.5));
  int64_t const row_start = int64_t(i) * (2 * int64_t(n) - int64_t(i) - 1) / 2;
  int const j = int(int64_t(k) - row_start + int64_t(i) + 1);
  return make_tuple(i, j);
}

}  // namespace common
}  // namespace score
}  // namespace tmol
