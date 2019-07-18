#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <ATen/Tensor.h>

#include "params.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct RamaDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<RamaParameters<Int>, 2, D> params,
      TView<Real, 3, D> tables,
      TView<RamaTableParams<Real>, 1, D> table_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D> >;
};

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
