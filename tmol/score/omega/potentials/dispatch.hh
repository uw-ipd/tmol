#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <ATen/Tensor.h>
#include "params.hh"

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct OmegaDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<OmegaParameters<Real>, 2, D> omega_indices)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D> >;
};

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
