#pragma once

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
struct GeneratePoseHydrogens {
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> coords,
      TView<Int, 3, Dev> h_coords_missing,
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

      TView<Int, 4, Dev> block_type_atom_ancestors,  // n-bt x max-n-ats x 4 x 3

      TView<Real, 3, Dev>
          block_type_atom_icoors  // n-bt x max-n-ats x 3 [phi, theta, D]

      ) -> TPack<Vec<Real, 3>, 2, Dev>;

  static auto backward(
      TView<Vec<Real, 3>, 2, Dev> dE_d_new_coords,
      TView<Vec<Real, 3>, 2, Dev> orig_coords,
      TView<Int, 3, Dev> h_coords_missing,
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

      TView<Int, 4, Dev> block_type_atom_ancestors,  // n-bt x max-n-ats x 4 x 3

      TView<Real, 3, Dev>
          block_type_atom_icoors  // n-bt x max-n-ats x 3 [phi, theta, D]
      ) -> TPack<Vec<Real, 3>, 2, Dev>;
};

#undef def

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
