#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include "params.hh"

#undef B0

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class SingleDispatch,
    template <tmol::Device>
    class PairDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct HBondDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, Dev> donor_coords,
      TView<Vec<Real, 3>, 1, Dev> acceptor_coords,

      TView<int64_t, 1, Dev> D,
      TView<int64_t, 1, Dev> H,
      TView<Int, 1, Dev> donor_type,

      TView<int64_t, 1, Dev> A,
      TView<int64_t, 1, Dev> B,
      TView<int64_t, 1, Dev> B0,
      TView<Int, 1, Dev> acceptor_type,

      TView<HBondPairParams<Real>, 2, Dev> pair_params,
      TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
      TView<HBondGlobalParams<Real>, 1, Dev> global_params)
      -> std::tuple<
          TPack<Real, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>,
          TPack<Vec<Real, 3>, 1, Dev>>;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
