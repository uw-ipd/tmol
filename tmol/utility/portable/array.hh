#pragma once

#include "qualifiers.hh"
#include "tuple.hh"

// Fixed-size array, standing in for std::array in device code.
//
// tmol's multi-term tile evaluators carry one score per term in a std::array
// and unpack it with std::get. <array> is unavailable in Metal, so this gives
// the same aggregate-initializable shape and the same get<I> spelling.

namespace tmol {
namespace portable {

template <typename T, int N>
struct array {
  T elems[N];

  TMOL_HD TMOL_THREAD T& operator[](int i) { return elems[i]; }
  TMOL_HD TMOL_THREAD T const& operator[](int i) const {
    return elems[i];
  }
  TMOL_HD static constexpr int size() { return N; }
};

// Zero-length arrays appear where a tile evaluator is instantiated with its
// scores disabled. A zero-size member array is not standard C++, so the empty
// case is spelled separately.
template <typename T>
struct array<T, 0> {
  TMOL_HD static constexpr int size() { return 0; }
};

template <int I, typename T, int N>
TMOL_HD TMOL_THREAD T& get(TMOL_THREAD array<T, N>& a) {
  return a.elems[I];
}

template <int I, typename T, int N>
TMOL_HD TMOL_THREAD T const& get(TMOL_THREAD array<T, N> const& a) {
  return a.elems[I];
}

}  // namespace portable
}  // namespace tmol
