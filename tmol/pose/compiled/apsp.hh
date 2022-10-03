#pragma once

// #include <Eigen/Core>
// #include <tuple>

#include <torch/torch.h>

// #include <pybind11/eigen.h>

#include <tmol/utility/tensor/TensorAccessor.h>
// #include <tmol/utility/tensor/TensorPack.h>
// #include <tmol/utility/tensor/pybind.h>
// #include <tmol/utility/function_dispatch/pybind.hh>

namespace tmol {
namespace pose {

// compute the all-pairs-shortest paths for each graph
// with the "weights" (distances)
template <tmol::Device D, typename Int>
struct AllPairsShortestPathsDispatch {
  static void f(TView<Int, 3, D> weights);
};

// compute the all-pairs shortest paths up to a given
// distance limit; all pairs with distances greater
// than the limit are approximated as being at that limit

// template <tmol::Device D, typename Int>
// struct LimitedSparseAllPairsShortestPathsDispatch {
//
//   static
//   TPack<Int, 3, D>
//   f(
//     TView<Int, 2, D> n_conn_for_nodes,
//     TView<Int, 2, D> conn_offset_for_nodes,
//     TView<Int, 3, D> connections_for_nodes,
//     int limit
//   );
//
// };

}  // namespace pose
}  // namespace tmol
