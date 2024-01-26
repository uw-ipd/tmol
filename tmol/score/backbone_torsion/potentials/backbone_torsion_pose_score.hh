#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/unresolved_atom.hh>
#include <tmol/score/common/tuple.hh>

#include <tmol/score/backbone_torsion/potentials/params.hh>

namespace tmol {
namespace score {
namespace backbone_torsion {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
class BackboneTorsionPoseScoreDispatch {
 public:
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // n_block_types x max_n_atoms_per_block_type
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // [n_block_types x 2]; -1 if rama not defined for a given block type
      // For second dim, 0 if upper neighbor is not proline
      // 1 if upper neighbor is proline
      TView<Int, 2, Dev> block_type_rama_table,

      // [n_block_types x 2]: same as block_type_rama_table
      TView<Int, 2, Dev> block_type_omega_table,

      // [n_block_types]: -1 if rama no upper connection exists
      TView<Int, 1, Dev> block_type_upper_conn_ind,
      // [n_block_types]: 1 if the bt is proline, 0 ow
      TView<Int, 1, Dev> block_type_is_pro,

      // n_block_types x 12
      // The 12 atoms that define the three torsions for every block type
      TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,
      //////////////////////

      // Rama potential parameters
      TView<Real, 3, Dev> rama_tables,
      TView<RamaTableParams<Real>, 1, Dev> rama_table_params,

      // Omega (backbone-dependent) potential parameters
      TView<Real, 4, Dev> omega_tables,
      TView<RamaTableParams<Real>, 1, Dev> omega_table_params,
      bool output_block_pair_energies)
      -> std::tuple<TPack<Real, 4, Dev>, TPack<Vec<Real, 3>, 3, Dev>>;

  static auto backward(
      TView<Vec<Real, 3>, 2, Dev> coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // n_block_types x max_n_atoms_per_block_type
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // [n_block_types x 2]; -1 if rama not defined for a given block type
      // For second dim, 0 if upper neighbor is not proline
      // 1 if upper neighbor is proline
      TView<Int, 2, Dev> block_type_rama_table,

      // [n_block_types x 2]: same as block_type_rama_table
      TView<Int, 2, Dev> block_type_omega_table,

      // [n_block_types]: -1 if rama no upper connection exists
      TView<Int, 1, Dev> block_type_upper_conn_ind,
      // [n_block_types]: 1 if the bt is proline, 0 ow
      TView<Int, 1, Dev> block_type_is_pro,

      // n_block_types x 12
      // The 12 atoms that define the three torsions for every block type
      TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,
      //////////////////////

      // Rama potential parameters
      TView<Real, 3, Dev> rama_tables,
      TView<RamaTableParams<Real>, 1, Dev> rama_table_params,

      // Omega (backbone-dependent) potential parameters
      TView<Real, 4, Dev> omega_tables,
      TView<RamaTableParams<Real>, 1, Dev> omega_table_params,
      TView<Real, 4, Dev> dTdV  // nterms x nposes x (1|len) x (1|len)
      ) -> TPack<Vec<Real, 3>, 3, Dev>;
};

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
