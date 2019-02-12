#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <tmol::Device D, typename Real, typename Int>
struct CartBondedLengthDispatch {
  static auto f(
      TView<Vec<Int, 2>, 1, D> atompair_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 1, D> K,
      TView<Real, 1, D> x0)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> >;
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedAngleDispatch {
  static auto f(
      TView<Vec<Int, 3>, 1, D> atomtriple_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 1, D> K,
      TView<Real, 1, D> x0)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> >;
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedTorsionDispatch {
  static auto f(
      TView<Vec<Int, 4>, 1, D> atomquad_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 1, D> K,
      TView<Real, 1, D> x0,
      TView<Int, 1, D> period)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> >;
};

template <tmol::Device D, typename Real, typename Int>
struct CartBondedHxlTorsionDispatch {
  static auto f(
      TView<Vec<Int, 4>, 1, D> atomquad_indices,
      TView<Int, 1, D> parameter_indices,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 1, D> K1,
      TView<Real, 1, D> K2,
      TView<Real, 1, D> K3,
      TView<Real, 1, D> phi1,
      TView<Real, 1, D> phi2,
      TView<Real, 1, D> phi3)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> >;
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
