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
namespace dunbrack {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Coord Vec<Real, 3>

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    int MAXBB,
    int MAXCHI>
struct DunbrackDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, MAXBB + 1, D> rotameric_tables,
      TView<RotamericTableParams<Real, MAXBB>, 1, D> rotameric_table_params,
      TView<Real, MAXBB + 2, D> semirotameric_tables,
      TView<SemirotamericTableParams<Real, MAXBB>, 1, D>
          semirotameric_table_params,
      TView<DunResParameters<Int, MAXBB, MAXCHI>, 1, D> residue_params,
      TView<DunTableLookupParams<Int, MAXCHI>, 1, D> residue_lookup_params)
      -> std::tuple<TPack<Real, 1, D>, TPack<Coord, 2, D> >;
};

#undef Coord

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
