#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>

#include "lj.hh"
#include "lk_isotropic.hh"
#include "params.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

using std::tie;
using std::tuple;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct LJDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Int, 1, D> atom_type_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<Int, 1, D> atom_type_j,

      TView<Real, 2, D> bonded_path_lengths,
      LJTypeParamTensors<Real, D> type_params,
      LJGlobalParams<Real> global_params)
      -> std::tuple<
          TPack<int64_t, 2, D>,
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> >;
};

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct LKIsotropicDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> coords_i,
      TView<Int, 1, D> atom_type_i,

      TView<Vec<Real, 3>, 1, D> coords_j,
      TView<Int, 1, D> atom_type_j,

      TView<Real, 2, D> bonded_path_lengths,

      LKTypeParamTensors<Real, D> type_params,
      LJGlobalParams<Real> global_params)
      -> std::tuple<
          TPack<int64_t, 2, D>,
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 1, D>,
          TPack<Vec<Real, 3>, 1, D> >;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
