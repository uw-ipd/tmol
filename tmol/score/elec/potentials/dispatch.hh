#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

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
    TView<Vec<Real, 3>, 1, Dev> x_i,
    TView<Real, 1, Dev> e_i,
    TView<Vec<Real, 3>, 1, Dev> x_j,
    TView<Real, 1, Dev> e_j,
    TView<Real, 2, Dev> bonded_path_lengths,
    Real D,
    Real D0,
    Real S,
    Real min_dis,
    Real max_dis)
    -> std::tuple<
          TPack<int64_t, 2, Dev>,
          TPack<Real, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>>;
};

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
