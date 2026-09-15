#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real, int N>
using GridLKBVec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct GridLKBallDispatch {
  static auto build_waters(
      ContextManager& mgr,
      TView<GridLKBVec<Real, 3>, 2, D> ligand_coords,
      TView<Int, 1, D> ligand_polar_atoms,
      TView<Int, 2, D> ligand_water_kinds,
      TView<Int, 3, D> ligand_water_atoms,
      TView<GridLKBVec<Real, 3>, 2, D> ligand_water_params,
      TView<Int, 2, D> ligand_water_is_ring,
      Real water_distance) -> TPack<GridLKBVec<Real, 3>, 3, D>;

  static auto propagate_water_gradients(
      ContextManager& mgr,
      TView<GridLKBVec<Real, 3>, 2, D> ligand_coords,
      TView<Int, 1, D> ligand_polar_atoms,
      TView<Int, 2, D> ligand_water_kinds,
      TView<Int, 3, D> ligand_water_atoms,
      TView<GridLKBVec<Real, 3>, 2, D> ligand_water_params,
      TView<Int, 2, D> ligand_water_is_ring,
      TView<GridLKBVec<Real, 3>, 3, D> water_gradients,
      Real water_distance) -> TPack<GridLKBVec<Real, 3>, 2, D>;
};

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
