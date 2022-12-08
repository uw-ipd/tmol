#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/elec/potentials/params.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct ElecPoseScoreDispatch {
  static auto f(
      TView<Vec<Real, 3>, 2, D> coords,
      TView<Int, 2, D> pose_stack_block_coord_offset,
      TView<Int, 2, D> pose_stack_block_type,

      // dims: n-poses x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
      TView<Int, 3, D> pose_stack_min_bond_separation,

      // dims: n-poses x max-n-blocks x max-n-blocks x
      // max-n-interblock-connections x max-n-interblock-connections
      TView<Int, 5, D> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, D> block_type_n_atoms,

      TView<Real, 2, D> block_type_partial_charge,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, D> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

      // what is the path distance between pairs of atoms in the block
      // denormalized by their count-pair representative; used for
      // inter-block chemical-bond separation determination. Entry
      // i, j stores path_dist[i, rep(j)]
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, D> block_type_inter_repr_path_distance,

      // what is the path distance between pairs of atoms in the block
      // denormalized (twice!) by their count-pair representative;
      // used for intra-block chemical-bond separation determination.
      // Entry i, j stores path_dist[rep(i), rep(j)]
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, D> block_type_intra_repr_path_distance,
      //////////////////////

      // LJ parameters
      TView<ElecGlobalParams<Real>, 1, D> global_params,
      bool compute_derivs)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>>;
};

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
