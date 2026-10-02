#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/hbond/potentials/params.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct HBondPoseScoreDispatch {
  static auto forward(
      ContextManager& mgr,
      // common params
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_donor_type,
      TView<Int, 3, Dev> block_type_tile_acceptor_type,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      //////////////////////

      // HBond potential parameters
      TView<HBondPairParams<Real>, 2, Dev> pair_params,
      TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
      TView<HBondGlobalParams<Real>, 1, Dev> global_params,

      // Derived-atom coords + source-atom indices produced by the
      // hbond pre-pass kernel (GenerateHBondBases).
      TView<Vec<Real, 3>, 2, Dev> derived_coords,
      TView<Int, 2, Dev> derived_atom_inds,

      TView<Int, 1, Dev> shared_compact_block_neighbors,
      bool output_block_pair_energies,
      bool compute_derivs,
      bool allow_split_pairs = false)
      -> std::tuple<
          TPack<Real, 4, Dev>,
          TPack<Vec<Real, 3>, 2, Dev>,
          TPack<Int, 3, Dev>>;

  static auto backward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_donor_type,
      TView<Int, 3, Dev> block_type_tile_acceptor_type,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      //////////////////////

      // HBond potential parameters
      TView<HBondPairParams<Real>, 2, Dev> pair_params,
      TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
      TView<HBondGlobalParams<Real>, 1, Dev> global_params,
      //////////////////////

      // Derived-atom coords + source-atom indices produced by the
      // hbond pre-pass kernel (GenerateHBondBases).
      TView<Vec<Real, 3>, 2, Dev> derived_coords,
      TView<Int, 2, Dev> derived_atom_inds,

      TView<Int, 3, Dev> block_neighbors,  // from forward pass
      TView<Real, 4, Dev> dTdV             // nterms x nposes x len x len
      ) -> TPack<Vec<Real, 3>, 1, Dev>;
};

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct HBondRotamerScoreDispatch {
  static auto rotamer_spheres(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      TView<Int, 1, Dev> block_type_n_atoms)
      -> std::tuple<TPack<Real, 2, Dev>, TPack<Real, 3, Dev>>;

  static auto rotamer_dispatch_page(
      ContextManager& mgr,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Real, 3, Dev> block_spheres,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      TView<Real, 2, Dev> rot_spheres,
      TView<Int, 2, Dev> lockstep_group_for_block,
      Real reach,
      Int candidate_begin,
      Int candidate_end) -> TPack<Int, 2, Dev>;

  static auto forward(
      ContextManager& mgr,
      // common params
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
      // lockstep, so only matching rotamer indices ever coexist
      TView<Int, 2, Dev> lockstep_group_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_donor_type,
      TView<Int, 3, Dev> block_type_tile_acceptor_type,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      //////////////////////

      // HBond potential parameters
      TView<HBondPairParams<Real>, 2, Dev> pair_params,
      TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
      TView<HBondGlobalParams<Real>, 1, Dev> global_params,

      // Derived-atom coords + source-atom indices produced by the
      // hbond pre-pass kernel (GenerateHBondBases).
      TView<Vec<Real, 3>, 2, Dev> derived_coords,
      TView<Int, 2, Dev> derived_atom_inds,

      bool output_block_pair_energies,
      bool compute_derivs,
      TPack<Int, 2, Dev> shared_dispatch_indices)
      -> std::tuple<
          TPack<Real, 2, Dev>,
          TPack<Vec<Real, 3>, 2, Dev>,
          TPack<Int, 2, Dev>>;

  static auto backward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
      // and [pose, block1, slot, conn1, conn2]
      TView<Int, 4, Dev> pose_stack_near_blocks,
      TView<int8_t, 5, Dev> pose_stack_inter_block_bondsep,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,  // ?? needed ?? I think so

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,
      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_donor_type,
      TView<Int, 3, Dev> block_type_tile_acceptor_type,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      //////////////////////

      // HBond potential parameters
      TView<HBondPairParams<Real>, 2, Dev> pair_params,
      TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
      TView<HBondGlobalParams<Real>, 1, Dev> global_params,
      //////////////////////

      // Derived-atom coords + source-atom indices produced by the
      // hbond pre-pass kernel (GenerateHBondBases).
      TView<Vec<Real, 3>, 2, Dev> derived_coords,
      TView<Int, 2, Dev> derived_atom_inds,

      TView<Int, 2, Dev> dispatch_indices,  // from forward pass
      TView<Real, 2, Dev> dTdV              // nterms x nposes x len x len
      ) -> TPack<Vec<Real, 3>, 1, Dev>;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
