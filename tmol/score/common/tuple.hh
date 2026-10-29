#pragma once

#include <tmol/utility/portable/tuple.hh>

namespace tmol {
namespace score {
namespace common {

// Multiple-return values from potentials use the portable tuple so the same
// headers compile for the host, nvcc, and Metal. astuple(), which exists only
// to hand results to pybind, keeps std::tuple explicitly; it is a member of a
// class template and so is never instantiated on a backend that cannot use it.
using tmol::portable::get;
using tmol::portable::index_sequence;
using tmol::portable::index_sequence_for;
using tmol::portable::make_index_sequence;
using tmol::portable::make_tuple;
using tmol::portable::tie;
using tmol::portable::tuple;

}  // namespace common
}  // namespace score
}  // namespace tmol
