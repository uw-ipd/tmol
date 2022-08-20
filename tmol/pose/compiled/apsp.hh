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

}  // namespace pose
}  // namespace tmol
