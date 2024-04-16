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

#include <tmol/score/hbond/potentials/params.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct HBondPoseScoreDispatch {
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // dims: n-poses x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
      TView<Int, 3, Dev>
          pose_stack_min_bond_separation,  // ?? needed ?? I think so

      // dims: n-poses x max-n-blocks x max-n-blocks x
      // max-n-interblock-connections x max-n-interblock-connections
      TView<Int, 5, Dev>
          pose_stack_inter_block_bondsep,  // ?? needed ?? I think so

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

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_donor_type,
      TView<Int, 3, Dev> block_type_tile_acceptor_type,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // HBond potential parameters
      TView<HBondPairParams<Real>, 2, Dev> pair_params,
      TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
      TView<HBondGlobalParams<Real>, 1, Dev> global_params,

      bool output_block_pair_energies,
      bool compute_derivs)
      -> std::tuple<
          TPack<Real, 4, Dev>,
          TPack<Vec<Real, 3>, 3, Dev>,
          TPack<Int, 3, Dev> >;

  static auto backward(
      TView<Vec<Real, 3>, 2, Dev> coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      // dims: n-poses x max-n-blocks x max-n-blocks
      // Quick lookup: given the inds of two blocks, ask: what is the minimum
      // number of chemical bonds that separate any pair of atoms in those
      // blocks? If this minimum is greater than the crossover, then no further
      // logic for deciding whether two atoms in those blocks should have their
      // interaction energies calculated: all should. intentionally small to
      // (possibly) fit in constant cache
      TView<Int, 3, Dev>
          pose_stack_min_bond_separation,  // ?? needed ?? I think so

      // dims: n-poses x max-n-blocks x max-n-blocks x
      // max-n-interblock-connections x max-n-interblock-connections
      TView<Int, 5, Dev>
          pose_stack_inter_block_bondsep,  // ?? needed ?? I think so

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

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_donor_type,
      TView<Int, 3, Dev> block_type_tile_acceptor_type,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      // How many chemical bonds separate all pairs of atoms
      // within each block type?
      // Dimsize: n_block_types x max_n_atoms x max_n_atoms
      TView<Int, 3, Dev> block_type_path_distance,

      //////////////////////

      // HBond potential parameters
      TView<HBondPairParams<Real>, 2, Dev> pair_params,
      TView<HBondPolynomials<double>, 2, Dev> pair_polynomials,
      TView<HBondGlobalParams<Real>, 1, Dev> global_params,

      TView<Int, 3, Dev> scratch_block_neighbors,  // from forward pass
      TView<Real, 4, Dev> dTdV  // nterms x nposes x len x len
      ) -> TPack<Vec<Real, 3>, 3, Dev>;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
