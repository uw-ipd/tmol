#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// Per-pose pre-pass kernel that, for every atom in the pose, computes
// the coordinates of the "derived atoms" needed by the hbond pairwise
// scoring kernel.  Outputs two tensors, both indexed by the global
// pose-atom index that already addresses rot_coords:
//
//   derived_coords [n_pose_atoms, 3]  : Real3
//     slot 0: D  (donor heavy atom)  for atoms that are donor hydrogens
//     slot 1: B  (acceptor base)     for atoms that are acceptors
//     slot 2: B0 (acceptor base 2)   for atoms that are acceptors
//
//   derived_atom_inds [n_pose_atoms, 3] : Int
//     The global pose-atom index of the source atom (the atom whose
//     position was copied into the corresponding derived_coords slot).
//     The pairwise kernel uses this to attribute gradients back to the
//     source atom without re-walking the bond graph.
//     Set to -1 for unused slots.
//
// Atoms that are neither donor H nor acceptor have all three slots
// NaN / -1 and are never read by the pairwise kernel.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct GenerateHBondBases {
  static auto forward(
      ContextManager& mgr,
      // common pose params
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected to each other
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      //////////////////////
      // Chemical properties
      TView<Int, 1, Dev> block_type_n_atoms,
      TView<Int, 1, Dev> block_type_n_interblock_bonds,
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      // Bond-graph topology, used to identify acceptor base / donor
      // heavy atoms via bond traversal
      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

      // hbond per-tile precomputed atom lists (already produced for the
      // pairwise kernel; reused unchanged here)
      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen)
      -> std::tuple<TPack<Vec<Real, 3>, 2, Dev>, TPack<Int, 2, Dev> >;
};

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
