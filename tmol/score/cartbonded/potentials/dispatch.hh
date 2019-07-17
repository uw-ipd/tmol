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

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<CartBondedLengthParams<Int>, 1, D> cbl_atoms,
      TView<CartBondedAngleParams<Int>, 1, D> cba_atoms,
      TView<CartBondedTorsionParams<Int>, 1, D> cbt_atoms,
      TView<CartBondedTorsionParams<Int>, 1, D> cbi_atoms,
      TView<CartBondedTorsionParams<Int>, 1, D> cbhxl_atoms,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> cbl_params,
      TView<CartBondedHarmonicTypeParams<Real>, 1, D> cba_params,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> cbt_params,
      TView<CartBondedPeriodicTypeParams<Real>, 1, D> cbi_params,
      TView<CartBondedSinusoidalTypeParams<Real>, 1, D> cbhxl_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D> >;
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
