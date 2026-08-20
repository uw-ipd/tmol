#pragma once

#include <cstdint>

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct GridInterpDispatch {
  static auto forward(
      ContextManager& mgr,
      TView<Real, 4, D> coefficients,
      TView<Vec<Real, 3>, 1, D> indices,
      TView<Int, 1, D> grid_indices) -> TPack<Real, 1, D>;
};

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
