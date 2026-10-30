#pragma once

#include <tmol/score/elec/potentials/potentials.hh>
#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real>
struct GridBuildLKWeights {
  Real iso;
  Real ball;
  Real bridge;
  Real bridge_uncoupled;
};

// Receptor-vs-probe energy grids on one device. For every (probe type, voxel):
// the LJ / LK / (optional) LK-ball sums over the receptor atoms in the 27 cells
// around the voxel; and the probe-independent electrostatics per voxel.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct GridBuildDispatch {
  static void build(
      ContextManager& mgr,
      TView<Real, 2, D> receptor_coords,
      TView<Int, 1, D> receptor_types,
      TView<Real, 1, D> receptor_charges,
      TView<Real, 3, D> receptor_waters,  // [n, 4, 3]; read only for lk_ball
      TView<Int, 1, D> probe_types,
      TView<Real, 2, D> type_params,  // [type, 10] raw ljlk parameters
      tmol::score::ljlk::potentials::LJGlobalParams<Real> lj_global,
      tmol::score::elec::potentials::ElecGlobalParams<Real> elec_global,
      GridBuildLKWeights<Real> lk_weights,
      TView<Int, 1, D> cell_offsets,  // CSR over receptor atoms by cell
      TView<Int, 1, D> cell_indices,
      TView<Real, 1, D> cell_origin,  // cells are max-cutoff wide
      TView<Int, 1, D> cell_dims,
      TView<Real, 1, D> grid_origin,
      Int n_x,
      Int n_y,
      Int n_z,
      Real voxel_spacing,
      bool build_lk_ball,
      TView<Real, 2, D> atr,  // [probe, voxel]
      TView<Real, 2, D> rep,
      TView<Real, 2, D> sol,
      TView<Real, 2, D> lk_ball,
      TView<Real, 2, D> lk_water,
      TView<Real, 2, D> lk_bridge,
      TView<Real, 1, D> elec);  // [voxel]
};

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
