#pragma once

#include <tmol/utility/tensor/pybind.h>
#include "bonded_atom.hh"

namespace pybind11 {
namespace detail {

#define CAST_ATTR(SRC, TARGET, NAME)                             \
  try {                                                          \
    TARGET.NAME = SRC.attr(#NAME).cast<decltype(TARGET.NAME)>(); \
  } catch (pybind11::cast_error) {                               \
    pybind11::print("Error casting: ", #NAME);                   \
    return false;                                                \
  }

template <typename Int, tmol::Device D>
struct type_caster<tmol::score::bonded_atom::IndexedBonds<Int, D>> {
 public:
  typedef tmol::score::hbond::IndexedBonds<Int, D> T;

  PYBIND11_TYPE_CASTER(T, _<T>());

  bool load(handle src, bool convert) {
    CAST_ATTR(src, value, bonds);
    CAST_ATTR(src, value, bond_spans);

    return true;
  }
};

}  // namespace detail
}  // namespace pybind11
#undef CAST_ATTR
