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

#include <tmol/score/backbone_torsion/potentials/params.hh>
#include <tmol/score/backbone_torsion/potentials/potentials.hh>
#include <tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.hh>

#include <chrono>

namespace tmol {
namespace score {
namespace backbone_torsion {
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
auto BackboneTorsionPoseScoreDispatch<DeviceDispatch, Dev, Real, Int>::f(
    TView<Vec<Real, 3>, 2, Dev> coords,
    TView<Int, 2, Dev> pose_stack_block_coord_offset,
    TView<Int, 2, Dev> pose_stack_block_type,

    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,

    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params)
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
  int const n_omega_tables = omega_tables.size(0);

  assert(pose_stack_block_coord_offset.size(0) == n_poses);
  assert(pose_stack_block_coord_offset.size(1) == max_n_blocks);

  assert(pose_stack_block_type.size(0) == n_poses);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

  assert(block_type_rama_table.size(0) == n_block_types);
  assert(block_type_omega_table.size(0) == n_block_types);

  assert(block_type_torsion_atoms.size(0) == n_block_types);
  assert(block_type_torsion_atoms.size(1) == 12);

  assert(rama_table_params.size(0) == n_rama_tables);
  assert(omega_table_params.size(0) == n_omega_tables);

  auto V_t = TPack<Real, 2, Dev>::zeros({2, n_poses});
  auto dV_dxyz_t =
      TPack<Vec<Real, 3>, 3, Dev>::zeros({2, n_poses, max_n_pose_atoms});

  auto V = V_t.view;
  auto dV_dxyz = dV_dxyz_t.view;

  LAUNCH_BOX_32;
  // Define nt
  CTA_LAUNCH_T_PARAMS;

  auto rama_omega_func = ([=] TMOL_DEVICE_FUNC(int ind) {
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
        pose_stack_inter_block_connections[pose_ind][block_ind][upper_conn][0];
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

    bool valid_torsion = true;
    Vec<Int, 4> phi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type][i];
      phi_ats[i] = resolve_atom_from_uaid(
          i_at,
          block_ind,
          pose_ind,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_torsion &= phi_ats[i] != -1;
    }
    CoordQuad<Real> phi_coords;
    for (int i = 0; i < 4; ++i) {
      phi_coords.row(i) = coords[pose_ind][phi_ats[i]];
    }
    Vec<Int, 4> psi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type][i + 4];
      psi_ats[i] = resolve_atom_from_uaid(
          i_at,
          block_ind,
          pose_ind,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_torsion &= psi_ats[i] != -1;
    }
    CoordQuad<Real> psi_coords;
    for (int i = 0; i < 4; ++i) {
      psi_coords.row(i) = coords[pose_ind][psi_ats[i]];
    }

    // accumulate rama
    if (valid_torsion) {
      auto rama = rama_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          rama_tables[rama_table_ind],
          Eigen::Map<Vec<Real, 2>>(rama_table_params[rama_table_ind].bbstarts),
          Eigen::Map<Vec<Real, 2>>(rama_table_params[rama_table_ind].bbsteps));
      accumulate<Dev, Real>::add(V[0][pose_ind], common::get<0>(rama));
      for (int j = 0; j < 4; ++j) {
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][pose_ind][phi_ats[j]], common::get<1>(rama).row(j));
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][pose_ind][psi_ats[j]], common::get<2>(rama).row(j));
      }
    }

    CoordQuad<Real> omega_coords;
    Vec<Int, 4> omega_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type][i + 8];

      // Check to see if the omega uaids are actually defined for this block
      // type. If the atom offset [0] or the connection index [1] are both -1,
      // this is a sentinel for an undefined omega.
      if (i_at.atom_id == -1 && i_at.conn_id == -1)
        return;  // no omega and we already calculated rama

      omega_ats[i] = resolve_atom_from_uaid(
          i_at,
          block_ind,
          pose_ind,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);

      // omega_atom_ind == -1 -> UAID resolution failed
      if (omega_ats[i] == -1)
        return;  // no omega and we already calculated rama
    }

    for (int i = 0; i < 4; ++i) {
      omega_coords.row(i) = coords[pose_ind][omega_ats[i]];
    }

    // accumulate omega
    auto omega = omega_V_dV<Dev, Real, Int>(omega_coords, 32.8);
    printf("%f\n", common::get<0>(omega));
    accumulate<Dev, Real>::add(V[1][pose_ind], common::get<0>(omega));
    for (int j = 0; j < 4; ++j) {
      Vec<Real, 3> j_deriv = common::get<1>(omega).row(j);
      accumulate<Dev, Vec<Real, 3>>::add(
          dV_dxyz[1][pose_ind][omega_ats[j]], common::get<1>(omega).row(j));
    }
  });

  int n_blocks = n_poses * max_n_blocks;
  DeviceDispatch<Dev>::template forall<launch_t>(n_blocks, rama_omega_func);

  return {V_t, dV_dxyz_t};
};

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
