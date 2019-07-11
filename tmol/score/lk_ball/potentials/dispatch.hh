#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/dispatch.hh>

#include <tmol/score/ljlk/potentials/params.hh>

#include "params.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct LKBallDispatch {
  static auto forward(
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<int64_t, 1, D> polars_i,
      TView<Int, 1, D> atom_type_i,
      TView<Vec<Real, 3>, 2, D> waters_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<int64_t, 1, D> occluders_j,
      TView<Int, 1, D> atom_type_j,
      TView<Vec<Real, 3>, 2, D> waters_j,

      TView<Real, 2, D> bonded_path_lengths,

      TView<LKBallTypeParams<Real>, 1, D> type_params,
      TView<LKBallGlobalParams<Real>, 1, D> global_params) -> TPack<Real, 1, D>;

  static auto backward(
      TView<Real, 1, D> dTdV,
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<int64_t, 1, D> polars_i,
      TView<Int, 1, D> atom_type_i,
      TView<Vec<Real, 3>, 2, D> waters_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<int64_t, 1, D> occluders_j,
      TView<Int, 1, D> atom_type_j,
      TView<Vec<Real, 3>, 2, D> waters_j,

      TView<Real, 2, D> bonded_path_lengths,

      TView<LKBallTypeParams<Real>, 1, D> type_params,
      TView<LKBallGlobalParams<Real>, 1, D> global_params)
      -> std::tuple<
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 2, D>,
          TPack<Vec<Real, 3>, 2, D> >;
};

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
