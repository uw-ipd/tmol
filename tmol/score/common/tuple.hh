#pragma once

#ifdef __CUDACC__
#include <moderngpu/tuple.hxx>
#else
#include <tuple>
#endif

namespace tmol {
namespace score {
namespace common {

#ifdef __CUDACC__
using mgpu::get;
using mgpu::index_sequence_for;
using mgpu::integer_sequence;
using mgpu::make_index_sequence;
using mgpu::make_tuple;
using mgpu::tie;
using mgpu::tuple;

#else
using std::get;
using std::index_sequence_for;
using std::integer_sequence;
using std::make_index_sequence;
using std::make_tuple;
using std::tie;
using std::tuple;

#endif

}  // namespace common
}  // namespace score
}  // namespace tmol
