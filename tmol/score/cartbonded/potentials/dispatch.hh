#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/dispatch.hh>

#include "params.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedLengthDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedLengthParams<Int>, 1, D> atom_indices,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D> >;
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedAngleDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedAngleParams<Int>, 1, D> atom_indices,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D> >;
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedTorsionParams<Int>, 1, D> atom_indices,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D> >;
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedHxlTorsionDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedTorsionParams<Int>, 1, D> atom_indices,
      TView<CartBondedSinusoidalTypeParams<Real>, 1, D> param_table)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D> >;
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
