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

#define CoordQuad Eigen::Matrix<Real, 4, 3>

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct RamaDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Vec<Int, 4>, 1, D> phi_indices,
      TView<Vec<Int, 4>, 1, D> psi_indices,
      TView<Int, 1, D> parameter_indices,
      TCollection<Real, 2, D> tables,
      TView<Vec<Real, 2>, 1, D> bbstarts,
      TView<Vec<Real, 2>, 1, D> bbsteps) -> std::
      tuple<TPack<Real, 1, D>, TPack<CoordQuad, 1, D>, TPack<CoordQuad, 1, D> >;
};

#undef CoordQuad

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
