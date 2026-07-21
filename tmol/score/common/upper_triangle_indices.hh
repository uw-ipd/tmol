#pragma once

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
  int i = n - 2 - floor(sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
  int j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
  return make_tuple(i, j);
}

}  // namespace common
}  // namespace score
}  // namespace tmol
