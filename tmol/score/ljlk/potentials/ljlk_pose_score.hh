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

#include "lj.hh"
#include "params.hh"
// #include "rotamer_pair_energy_lj.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct LJLKPoseScoreDispatch {
  static auto forward(
      // common params
      TView<Vec<Real, 3>, 1, D> rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> pose_ind_for_atom,
      TView<Int, 2, D> first_rot_for_block,
      TView<Int, 2, D> first_rot_block_type,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // dims: n-systems x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
      TView<Int, 3, D> pose_stack_min_bond_separation,

      // dims: n-systems x max-n-blocks x max-n-blocks x
      // max-n-interblock-connections x max-n-interblock-connections
      TView<Int, 5, D> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, D> block_type_n_atoms,

      TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
      TView<Int, 2, D> block_type_heavy_atoms_in_tile,

      // what are the atom types for these atoms
      // Dimsize: n_block_types x max_n_atoms
      TView<Int, 2, D> block_type_atom_types,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, D> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

      // what is the path distance between pairs of atoms in the block
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, D> block_type_path_distance,
      //////////////////////

      // LJ parameters
      TView<LJLKTypeParams<Real>, 1, D> type_params,
      TView<LJGlobalParams<Real>, 1, D> global_params,

      // should the output be per-pose (npose x nterms x 1 x 1)
      //   or per block-pair (npose x nterms x len x len)
      bool output_block_pair_energies,

      // do we need to compute gradients?
      bool require_gradient) -> std::
      tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D> >;

  static auto backward(
      // common params
      TView<Vec<Real, 3>, 1, D> rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> pose_ind_for_atom,
      TView<Int, 2, D> first_rot_for_block,
      TView<Int, 2, D> first_rot_block_type,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // dims: n-systems x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
      TView<Int, 3, D> pose_stack_min_bond_separation,

      // dims: n-systems x max-n-blocks x max-n-blocks x
      // max-n-interblock-connections x max-n-interblock-connections
      TView<Int, 5, D> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, D> block_type_n_atoms,

      TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
      TView<Int, 2, D> block_type_heavy_atoms_in_tile,

      // what are the atom types for these atoms
      // Dimsize: n_block_types x max_n_atoms
      TView<Int, 2, D> block_type_atom_types,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, D> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

      // what is the path distance between pairs of atoms in the block
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, D> block_type_path_distance,
      //////////////////////

      // LJ parameters
      TView<LJLKTypeParams<Real>, 1, D> type_params,
      TView<LJGlobalParams<Real>, 1, D> global_params,

      TView<Int, 2, D> dispatch_indices,  // from forward pass
      TView<Real, 2, D> dTdV              // nterms x nposes x (1|len) x (1|len)
      ) -> TPack<Vec<Real, 3>, 2, D>;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
