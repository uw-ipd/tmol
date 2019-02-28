#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <ATen/Tensor.h>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <tmol::Device D, typename Real, typename Int>
struct RamaDispatch {
  static auto f(
      TCollection<Real, 2, D> tables,
      TView<Eigen::Matrix<Real, 2, 1>, 1, D> indices) -> TPack<Real, 1, D>;
};

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
