#pragma once

#include <tmol/utility/tensor/pybind.h>
#include "unresolved_atom.hh"

namespace pybind11 {
namespace detail {

template <typename Int>
struct npy_format_descriptor_name<tmol::UnresolvedAtomID<Int>> {
  static constexpr auto name =
      npy_format_descriptor_name<Int>::name + _("UnresolvedAtomID");
};

}  // namespace detail
}  // namespace pybind11
