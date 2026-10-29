#pragma once

#include <tuple>

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real, int N>
using GridVec = Eigen::Matrix<Real, N, 1>;

// Receptor-grid scoring of ligand poses.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct GridScoreDispatch {
  // Weighted fa_atr / fa_rep / fa_sol / fa_elec (+ the heavy-atom part of the
  // gridded LK-ball total when ``lk_ball`` is non-empty) and the out-of-bounds
  // penalty of every grid-scored atom: (energy [pose, grid atom], gradient
  // [pose, atom]) in Cartesian units; atoms off the grid get zero gradient.
  // One work item per (pose, grid atom), each with its own outputs.
  static auto atoms(
      ContextManager& mgr,
      TView<GridVec<Real, 3>, 2, D> coords,
      TView<Int, 1, D> grid_atoms,
      TView<Int, 1, D> atom_types,
      TView<Real, 1, D> charges,
      TView<Real, 4, D> atr,
      TView<Real, 4, D> rep,
      TView<Real, 4, D> sol,
      TView<Real, 4, D> elec,
      TView<Real, 4, D> lk_ball,
      TView<Real, 1, D> origin,
      Real voxel_spacing,
      TView<Real, 1, D> weights,
      Real out_of_bounds_penalty)
      -> std::tuple<TPack<Real, 2, D>, TPack<GridVec<Real, 3>, 2, D>>;

  // Rosetta's gridded LK-ball water terms: builds each polar atom's waters,
  // samples the water and bridge grids there, takes the softmax / softmin
  // over the waters and propagates the gradient back to the atoms. Adds to
  // ``energy`` and ``gradient`` in place (atomically on CUDA, serially within
  // a pose on CPU). One work item per (pose, polar atom).
  static void waters(
      ContextManager& mgr,
      TView<GridVec<Real, 3>, 2, D> coords,
      TView<Int, 1, D> polar_atoms,
      TView<Int, 2, D> water_kinds,
      TView<Int, 3, D> water_atoms,
      TView<GridVec<Real, 3>, 2, D> water_params,
      TView<Int, 2, D> water_is_ring,
      TView<Int, 1, D> polar_types,
      TView<Real, 4, D> water_grid,
      TView<Real, 4, D> bridge_grid,
      TView<Real, 1, D> origin,
      Real voxel_spacing,
      Real water_distance,
      Real fade,
      bool bridge_enabled,
      TView<Real, 1, D> energy,
      TView<GridVec<Real, 3>, 2, D> gradient);
};

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
