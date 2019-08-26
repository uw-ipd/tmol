#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include "params.hh"

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct ElecDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, Dev> coords_i,
      TView<Real, 2, Dev> e_i,
      TView<Vec<Real, 3>, 2, Dev> coords_j,
      TView<Real, 2, Dev> e_j,
      TView<Real, 3, Dev> bonded_path_lengths,
      TView<ElecGlobalParams<float>, 1, Dev> global_params)
      -> std::tuple<
          TPack<float, 1, Dev>,
          TPack<Vec<Real, 3>, 2, Dev>,
          TPack<Vec<Real, 3>, 2, Dev> >;
};

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
