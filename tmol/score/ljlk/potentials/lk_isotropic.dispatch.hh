#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/geom.hh>

#include "lk_isotropic.hh"
#include "params.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <template <tmol::Device> class Dispatch, tmol::Device D, typename Real, typename Int>
struct LKIsotropicDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, D> coords_i,
      TView<Int, 2, D> atom_type_i,

      TView<Vec<Real, 3>, 2, D> coords_j,
      TView<Int, 2, D> atom_type_j,

      TView<Real, 3, D> bonded_path_lengths,

      TView<LKTypeParams<Real>, 1, D> type_params,
      TView<LJGlobalParams<Real>, 1, D> global_params)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<Vec<Real, 3>, 2, D>,
          TPack<Vec<Real, 3>, 2, D> >;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
