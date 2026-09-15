#pragma once

#include <tuple>

#include <Eigen/Core>

#include <tmol/score/lk_ball/potentials/params.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template <typename Real, int N>
using LKBVec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct LiveLKBallDispatch {
  static auto forward(
      ContextManager& mgr,
      TView<LKBVec<Real, 3>, 2, D> ligand_coords,
      TView<lk_ball::potentials::LKBallTypeParams<Real>, 1, D>
          ligand_atom_params,
      TView<Int, 1, D> ligand_heavy_atoms,
      TView<Int, 1, D> ligand_polar_atoms,
      TView<Int, 1, D> ligand_atom_to_polar,
      TView<Int, 2, D> ligand_water_kinds,
      TView<Int, 3, D> ligand_water_atoms,
      TView<LKBVec<Real, 3>, 2, D> ligand_water_params,
      TView<Int, 2, D> ligand_water_is_ring,
      TView<LKBVec<Real, 3>, 1, D> receptor_coords,
      TView<lk_ball::potentials::LKBallTypeParams<Real>, 1, D>
          receptor_params,
      TView<LKBVec<Real, 3>, 2, D> receptor_waters,
      TView<Int, 1, D> receptor_polar_atoms,
      TView<Real, 1, D> cell_origin,
      TView<Int, 1, D> cell_dims,
      TView<Int, 1, D> receptor_cell_offsets,
      TView<Int, 1, D> receptor_cell_indices,
      TView<Int, 1, D> receptor_polar_cell_offsets,
      TView<Int, 1, D> receptor_polar_cell_indices,
      TView<lk_ball::potentials::LKBallGlobalParams<Real>, 1, D> global_params,
      bool use_cell_list)
      -> std::tuple<TPack<Real, 2, D>, TPack<LKBVec<Real, 3>, 3, D>>;
};

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
