#pragma once

#include <tmol/score/lk_ball/potentials/params.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct GeneratePoseWaters {
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> pose_coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_don_hvy_inds,
      TView<Int, 3, Dev> block_type_tile_which_donH_for_hvy,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 3, Dev> block_type_tile_acc_n_attached_H,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      TView<LKBallWaterGenGlobalParams<Real>, 1, Dev> global_params,
      TView<Real, 1, Dev> sp2_water_tors,
      TView<Real, 1, Dev> sp3_water_tors,
      TView<Real, 1, Dev> ring_water_tors) -> TPack<Vec<Real, 3>, 3, Dev>;

  static auto backward(
      TView<Vec<Real, 3>, 3, Dev> dE_dWxyz,
      TView<Vec<Real, 3>, 2, Dev> pose_coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_don_hvy_inds,
      TView<Int, 3, Dev> block_type_tile_which_donH_for_hvy,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 3, Dev> block_type_tile_acc_n_attached_H,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      TView<LKBallWaterGenGlobalParams<Real>, 1, Dev> global_params,
      TView<Real, 1, Dev> sp2_water_tors,
      TView<Real, 1, Dev> sp3_water_tors,
      TView<Real, 1, Dev> ring_water_tors) -> TPack<Vec<Real, 3>, 2, Dev>;
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
