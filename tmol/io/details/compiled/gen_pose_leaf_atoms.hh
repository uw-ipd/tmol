#pragma once

#include <tmol/score/unresolved_atom.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct GeneratePoseLeafAtoms {
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> coords,
      TView<bool, 3, Dev> orig_coords_atom_missing,
      TView<bool, 2, Dev> pose_stack_atom_missing,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // n-bt x max-n-ats x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors,

      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors,

      // TEMP! Handle the case when an atom's coordinate depends on
      // an un-resolvable atom, e.g., "down" for an N-terminal atom
      // n-bt x max-n-ats x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors_backup,
      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors_backup

      ) -> TPack<Vec<Real, 3>, 2, Dev>;

  static auto backward(
      TView<Vec<Real, 3>, 2, Dev> dE_d_new_coords,
      TView<Vec<Real, 3>, 2, Dev> new_coords,
      TView<Vec<Real, 3>, 2, Dev> orig_coords,
      TView<bool, 3, Dev> orig_coords_atom_missing,
      TView<bool, 2, Dev> pose_stack_atom_missing,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,
      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // n-bt x max-n-ats x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors,

      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors,

      // TEMP! Handle the case when an atom's coordinate depends on
      // an un-resolvable atom, e.g., "down" for an N-terminal atom
      // n-bt x max-n-ats x 3
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors_backup,
      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors_backup

      ) -> TPack<Vec<Real, 3>, 2, Dev>;
};

#undef def

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
