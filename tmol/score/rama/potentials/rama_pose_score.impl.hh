#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/unresolved_atom.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>

#include <tmol/score/rama/potentials/params.hh>
#include <tmol/score/rama/potentials/potentials.hh>

#include <chrono>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;
template <typename Real>
using CoordQuad = Eigen::Matrix<Real, 4, 3>;

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
class RamaPoseScoreDispatch {
 public:
  static auto f(
      TView<Vec<Real, 3>, 2, Dev> coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // n_block_types x max_n_conn x max_n_atoms_per_block_type
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // [n_block_types x 2]; -1 if rama not defined for a given block type
      // For second dim, 0 if upper neighbor is not proline
      // 1 if upper neighbor is proline
      TView<Int, 2, Dev> block_type_rama_table,
      // [n_block_types]: -1 if rama no upper connection exists
      TView<Int, 1, Dev> block_type_upper_conn_ind,
      // [n_block_types]: 1 if the bt is proline, 0 ow
      TView<Int, 1, Dev> block_type_is_pro,

      // n_block_types x 8
      // The 8 atoms that define the two torsions for every block type
      TView<UnresolvedAtomID<Int>, 2, Dev> block_type_rama_torsion_atoms,
      //////////////////////

      // Rama potential parameters
      TView<Real, 3, Dev> rama_tables,
      TView<RamaTableParams<Real>, 1, Dev> table_params)
      -> std::tuple<TPack<Real, 2, Dev>, TPack<Vec<Real, 3>, 3, Dev>> {
    using tmol::score::common::accumulate;
    using Real3 = Vec<Real, 3>;

    int const n_poses = coords.size(0);
    int const max_n_pose_atoms = coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_block_connections.size(2);
    int const n_block_types = block_type_atom_downstream_of_conn.size(0);
    int const max_n_atoms_per_block_type =
        block_type_atom_downstream_of_conn.size(1);
    int const n_rama_tables = rama_tables.size(0);

    assert(pose_stack_block_coord_offset.size(0) == n_poses);
    assert(pose_stack_block_coord_offset.size(1) == max_n_blocks);

    assert(pose_stack_block_type.size(0) == n_poses);

    assert(pose_stack_inter_block_connections.size(0) == n_poses);
    assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

    assert(block_type_rama_table.size(0) == n_block_types);

    assert(block_type_rama_torsion_atoms.size(0) == n_block_types);
    assert(block_type_rama_torsion_atoms.size(1) == 8);

    assert(table_params.size(0) == n_rama_tables);

    auto V_t = TPack<Real, 2, Dev>::zeros({1, n_poses});
    auto dV_dxyz_t =
        TPack<Vec<Real, 3>, 3, Dev>::zeros({1, n_poses, max_n_pose_atoms});

    auto V = V_t.view;
    auto dV_dxyz = dV_dxyz_t.view;

    LAUNCH_BOX_32;
    // Define nt
    CTA_LAUNCH_T_PARAMS;

    auto rama_func = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose_ind = ind / max_n_blocks;
      int const block_ind = ind % max_n_blocks;
      int const block_type = pose_stack_block_type[pose_ind][block_ind];
      if (block_type < 0) {
        return;
      }
      int const upper_conn = block_type_upper_conn_ind[block_type];
      if (upper_conn < 0) {
        return;
      }
      int const upper_nbr_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][upper_conn]
                                            [0];
      if (upper_nbr_ind < 0) {
        return;
      }
      int const upper_nbr_bt = pose_stack_block_type[pose_ind][upper_nbr_ind];
      if (upper_nbr_bt < 0) {
        return;
      }
      int const upper_nbr_is_pro = block_type_is_pro[upper_nbr_bt];
      int const rama_table_ind =
          block_type_rama_table[block_type][upper_nbr_is_pro];
      if (rama_table_ind < 0) {
        return;
      }

      int const offset_this_block_ind =
          pose_stack_block_coord_offset[pose_ind][block_ind];

      Vec<Int, 4> phi_ats;
      for (int i = 0; i < 4; ++i) {
        UnresolvedAtomID<Int> i_at =
            block_type_rama_torsion_atoms[block_type][i];
        phi_ats[i] = resolve_atom_from_uaid(
            i_at,
            block_ind,
            pose_ind,
            pose_stack_block_coord_offset,
            pose_stack_block_type,
            pose_stack_inter_block_connections,
            block_type_atom_downstream_of_conn);
        if (phi_ats[i] == -1) {
          return;
        }
      }
      CoordQuad<Real> phi_coords;
      for (int i = 0; i < 4; ++i) {
        phi_coords.row(i) = coords[pose_ind][phi_ats[i]];
      }
      Vec<Int, 4> psi_ats;
      for (int i = 0; i < 4; ++i) {
        UnresolvedAtomID<Int> i_at =
            block_type_rama_torsion_atoms[block_type][i + 4];
        psi_ats[i] = resolve_atom_from_uaid(
            i_at,
            block_ind,
            pose_ind,
            pose_stack_block_coord_offset,
            pose_stack_block_type,
            pose_stack_inter_block_connections,
            block_type_atom_downstream_of_conn);
        if (psi_ats[i] == -1) {
          return;
        }
      }
      CoordQuad<Real> psi_coords;
      for (int i = 0; i < 4; ++i) {
        psi_coords.row(i) = coords[pose_ind][psi_ats[i]];
      }
      auto rama = rama_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          rama_tables[rama_table_ind],
          Eigen::Map<Vec<Real, 2>>(table_params[rama_table_ind].bbstarts),
          Eigen::Map<Vec<Real, 2>>(table_params[rama_table_ind].bbsteps));

      accumulate<Dev, Real>::add(V[0][pose_ind], common::get<0>(rama));
      for (int j = 0; j < 4; ++j) {
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][pose_ind][phi_ats[j]], common::get<1>(rama).row(j));
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][pose_ind][psi_ats[j]], common::get<2>(rama).row(j));
      }
    });
    int n_blocks = n_poses * max_n_blocks;
    DeviceDispatch<Dev>::template forall<launch_t>(n_blocks, rama_func);

    return {V_t, dV_dxyz_t};
  }
};

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
