#pragma once

#include "qualifiers.hh"

namespace tmol {
namespace portable {

// Reduction operators and scan tags.
//
// These are the vocabulary that kernel bodies and the DeviceOperations
// interface speak: a body passes plus_t<Real>() to reduce_in_workgroup, and
// scan() is tagged exclusive or inclusive. moderngpu supplies equivalents, but
// its headers are CUDA-only, so naming them here keeps the shared bodies --
// and the CPU build, which has no business depending on a GPU library --
// independent of it. The CUDA backend maps these onto moderngpu internally.

template <typename T>
struct plus_t {
  TMOL_HD T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct maximum_t {
  TMOL_HD T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename T>
struct minimum_t {
  TMOL_HD T operator()(T a, T b) const { return a < b ? a : b; }
};

template <typename T>
struct less_t {
  TMOL_HD bool operator()(T a, T b) const { return a < b; }
};

enum class scan_type { exclusive, inclusive };

}  // namespace portable
}  // namespace tmol
