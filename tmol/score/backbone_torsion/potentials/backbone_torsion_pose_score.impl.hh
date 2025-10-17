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

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

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
auto BackboneTorsionPoseScoreDispatch<DeviceDispatch, Dev, Real, Int>::forward(
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

    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_lower_conn_ind,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,

    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params,
    bool output_block_pair_energies)
    -> std::tuple<TPack<Real, 2, Dev>, TPack<Vec<Real, 3>, 2, Dev>, TPack<Int, 2, Dev>> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conn = pose_stack_inter_block_connections.size(2);
  int const n_block_types = block_type_atom_downstream_of_conn.size(0);
  int const max_n_atoms_per_block_type =
      block_type_atom_downstream_of_conn.size(1);
  int const n_rama_tables = rama_tables.size(0);
  int const n_omega_tables = omega_tables.size(0);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

  assert(block_type_rama_table.size(0) == n_block_types);
  assert(block_type_omega_table.size(0) == n_block_types);
  assert(block_type_lower_conn_ind.size(0) == n_block_types);
  assert(block_type_upper_conn_ind.size(0) == n_block_types);

  assert(block_type_torsion_atoms.size(0) == n_block_types);
  assert(block_type_torsion_atoms.size(1) == 12);

  assert(rama_table_params.size(0) == n_rama_tables);
  assert(omega_table_params.size(0) == n_omega_tables);


  LAUNCH_BOX_32;
  // Define nt
  CTA_LAUNCH_T_PARAMS;

  // RamaPrePro is a pseudo-1body term that's really a 2-body term:
  // the score for residue i changes if residue i+1 becomes proline.
  // Thus we will first count the number
  // sum(i<N-1, n_rots[i]*n_rots[i+1]) + n_rots[N-1] for each Pose 
  // and allocate that many scores to cacluate, then calculate
  // them in the second step.

  auto n_energies_for_block_t = TPack<Int, 2, Dev>::zeros({n_poses, max_n_blocks});
  auto n_energies_for_block = n_energies_for_block_t.view;
  auto count_n_rotamer_energies = ([=] TMOL_DEVICE_FUNC (int index) {
    // Look at each residue and make sure that it has both upper and lower neighbors
    // and then write down the number of rotamer pair energies to evaluate 
    // as n_rots_i * n_rots_upper
    int const pose_ind = index / max_n_blocks;
    int const block_ind = index % max_n_blocks;

    int const pose_block_type = pose_stack_block_type[pose_ind][block_ind];
    if (pose_block_type == -1) {
      return;
    }
    int const n_rots = n_rots_for_block[pose_ind][block_ind];

    int const upper_conn = block_type_upper_conn_ind[pose_block_type];
    if (upper_conn < 0) {
      return;
    }
    int const lower_conn = block_type_upper_conn_ind[pose_block_type];
    if (lower_conn < 0) {
      return;
    }
    int const lower_nbr_block_ind =
        pose_stack_inter_block_connections[pose_ind][block_ind][lower_conn][0];
    if (lower_nbr_block_ind < 0) {
      // no lower neighbor --> no backbone torsion term
      return;
    }

    int const upper_nbr_block_ind =
        pose_stack_inter_block_connections[pose_ind][block_ind][upper_conn][0];
    if (upper_nbr_block_ind < 0) {
      return;
    }
    // the upper neighbor is a real residue: we will score
    // the n_rots_i x n_rots_j rotamers for this pair.
    // This will properly include upper neighbors for circular peptides,
    // too.
    int const upper_n_rots = n_rots_for_block[pose_ind][upper_nbr_block_ind]; 

    n_energies_for_block[pose_ind][block_ind] = n_rots * upper_n_rots;
  });
  DeviceDispatch<Dev>::template forall<launch_t>(n_poses * max_n_blocks, count_n_rotamer_energies);

  auto n_energies_for_block_offset_t = TPack<Int, 2, Dev>::zeros({n_poses, max_n_blocks});
  auto n_energies_for_block_offset = n_energies_for_block_offset_t.view;
  int n_dispatch_total =
      DeviceDispatch<Dev>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_energies_for_block.data(),
          n_energies_for_block_offset.data(),
          n_poses * max_n_blocks,
          mgpu::plus_t<Int>());

  TPack<Real, 2, Dev> V_t;
  auto dispatch_indices_t = TPack<Int, 2, Dev>::zeros({3, n_dispatch_total});
  if (output_block_pair_energies) {
    V_t = TPack<Real, 2, Dev>::zeros({2, n_dispatch_total});
  } else {
    V_t = TPack<Real, 2, Dev>::zeros({2, n_poses});
  }
  auto dV_dxyz_t = TPack<Vec<Real, 3>, 2, Dev>::zeros({2, n_atoms});

  int const max_n_rots_per_block = DeviceDispatch<Dev>::reduce(
    n_rots_for_block.data(), n_poses * max_n_blocks, mgpu::maximum_t<Int>()
  );



  auto V = V_t.view;
  auto dV_dxyz = dV_dxyz_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

  auto mark_dispatch_indices = ([=] TMOL_DEVICE_FUNC (int ind) {
    int const pose_ind = ind / (max_n_blocks * max_n_rots_per_block * max_n_rots_per_block);
    ind = ind - pose_ind * max_n_blocks * max_n_rots_per_block * max_n_rots_per_block;
    int const block1_ind = ind / (max_n_rots_per_block * max_n_rots_per_block);
    ind = ind - block1_ind * max_n_rots_per_block * max_n_rots_per_block;
    int const local_rot1_ind = ind / max_n_rots_per_block;
    int const local_rot2_ind = ind % max_n_rots_per_block;

    int const pose_block_type = pose_stack_block_type[pose_ind][block1_ind];
    if (pose_block_type == -1) {
      // Filter out blocks that are not real
      return;
    }
    int const n_rots1 = n_rots_for_block[pose_ind][block1_ind];
    if (local_rot1_ind > n_rots1) {
      return;
    }

    int const block1_sparse_dispatch_offset = n_energies_for_block_offset[pose_ind][block1_ind];
    int const block1_rot_offset = first_rot_for_block[pose_ind][block1_ind];

    bool block1_has_upper_neighbor = true;
    int const upper_conn = block_type_upper_conn_ind[pose_block_type];
    if (upper_conn < 0) {
      block1_has_upper_neighbor = false;
    } else {
      int const upper_nbr_block_ind =
          pose_stack_inter_block_connections[pose_ind][block1_ind][upper_conn][0];
      if (upper_nbr_block_ind != -1) {
        // the upper neighbor is a real residue: we will score
        // the n_rots_i x n_rots_j rotamers for this pair
        // This will properly include upper neighbors for circular peptides,
        // too, unless we have a two residue circular peptide which feels
        // chemically impossible.
        int const upper_n_rots = n_rots_for_block[pose_ind][upper_nbr_block_ind]; 

        if (local_rot2_ind > upper_n_rots) {
          return;
        }
        int const sparse_index = block1_sparse_dispatch_offset + local_rot1_ind * upper_n_rots + local_rot2_ind;
        int const rot1_ind = block1_rot_offset + local_rot1_ind;
        int const rot2_ind = first_rot_for_block[pose_ind][upper_nbr_block_ind] + local_rot2_ind;

        dispatch_indices[0][sparse_index] = pose_ind;
        dispatch_indices[1][sparse_index] = rot1_ind;
        dispatch_indices[2][sparse_index] = rot2_ind;

      }  
    }
  });
  DeviceDispatch<Dev>::template forall<launch_t>(
    n_poses * max_n_blocks * max_n_rots_per_block * max_n_rots_per_block,
    mark_dispatch_indices
  );


  auto rama_omega_func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = dispatch_indices[0][ind];

    int const rot_ind1 = dispatch_indices[1][ind];
    int const rot_ind2 = dispatch_indices[2][ind];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    // Where will we write the output?
    // In block-pair-scoring mode, we store one energy per rotamer;
    // In non-block-pair-scoring mode, we store one energy per pose
    // (using atomic_add to accumulate for each block/rotamer in the pose)
    int const V_ind = (output_block_pair_energies) ? ind : pose_ind;

    int const upper_nbr_is_pro = block_type_is_pro[block_type2];
    int const rama_table_ind =
        block_type_rama_table[block_type1][upper_nbr_is_pro];
    int const omega_table_ind =
        block_type_omega_table[block_type1][upper_nbr_is_pro];

    int const rot_offset1 = rot_coord_offset[rot_ind1];

    bool valid_phipsi = true;
    Vec<Int, 4> phi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i];
      phi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= phi_ats[i] != -1;
    }
    CoordQuad<Real> phi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        phi_coords.row(i) = rot_coords[phi_ats[i]];
      }
    }

    Vec<Int, 4> psi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 4];
      psi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= psi_ats[i] != -1;
    }
    CoordQuad<Real> psi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        psi_coords.row(i) = rot_coords[psi_ats[i]];
      }
    }

    // accumulate rama
    if (valid_phipsi && rama_table_ind >= 0) {
      auto rama = rama_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          rama_tables[rama_table_ind],
          Eigen::Map<Vec<Real, 2>>(rama_table_params[rama_table_ind].bbstarts),
          Eigen::Map<Vec<Real, 2>>(rama_table_params[rama_table_ind].bbsteps));
      accumulate<Dev, Real>::add(
          V[0][V_ind], common::get<0>(rama));
      for (int j = 0; j < 4; ++j) {
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][phi_ats[j]], common::get<1>(rama).row(j));
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][psi_ats[j]], common::get<2>(rama).row(j));
      }
    }

    if (omega_table_ind < 0) {
      return;
    }

    CoordQuad<Real> omega_coords;
    Vec<Int, 4> omega_ats;
    for (int i = 0; i < 4; ++i) {
      // TO DO: replace "8" in the lookup below
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 8];

      // Check to see if the omega uaids are actually defined for this block
      // type. If the atom offset [0] or the connection index [1] are both -1,
      // this is a sentinel for an undefined omega.
      if (i_at.atom_id == -1 && i_at.conn_id == -1)
        return;  // no omega and we already calculated rama

      omega_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);

      // omega_atom_ind == -1 -> UAID resolution failed
      if (omega_ats[i] == -1)
        return;  // no omega and we already calculated rama
    }

    for (int i = 0; i < 4; ++i) {
      omega_coords.row(i) = rot_coords[omega_ats[i]];
    }

    if (valid_phipsi && omega_table_ind >= 0) {
      auto omega = omega_bbdep_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          omega_coords,
          omega_tables[omega_table_ind][0],
          omega_tables[omega_table_ind][1],
          Eigen::Map<Vec<Real, 2>>(
              omega_table_params[omega_table_ind].bbstarts),
          Eigen::Map<Vec<Real, 2>>(omega_table_params[omega_table_ind].bbsteps),
          32.8);
      accumulate<Dev, Real>::add(V[1][V_ind], common::get<0>(omega));
      for (int j = 0; j < 4; ++j) {
        // omega : [V, dVdphi, dVdpsi, dVdomega]
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][phi_ats[j]], common::get<1>(omega).row(j));
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][psi_ats[j]], common::get<2>(omega).row(j));
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][omega_ats[j]], common::get<3>(omega).row(j));
      }
    } else {
      // if rama is undefined, fall back to old version
      auto omega = omega_V_dV<Dev, Real, Int>(omega_coords, 32.8);
      accumulate<Dev, Real>::add(V[1][V_ind], common::get<0>(omega));
      for (int j = 0; j < 4; ++j) {
        // omega : [V, dVdomega]
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][omega_ats[j]], common::get<1>(omega).row(j));
      }
    }
  });

  DeviceDispatch<Dev>::template forall<launch_t>(n_dispatch_total, rama_omega_func);

  return {V_t, dV_dxyz_t, dispatch_indices_t};
};

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int>
auto BackboneTorsionPoseScoreDispatch<DeviceDispatch, Dev, Real, Int>::backward(
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

    TView<Int, 2, Dev> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

    TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

    TView<Int, 2, Dev> block_type_rama_table,
    TView<Int, 2, Dev> block_type_omega_table,
    TView<Int, 1, Dev> block_type_lower_conn_ind,
    TView<Int, 1, Dev> block_type_upper_conn_ind,
    TView<Int, 1, Dev> block_type_is_pro,

    TView<UnresolvedAtomID<Int>, 2, Dev> block_type_torsion_atoms,

    TView<Real, 3, Dev> rama_tables,
    TView<RamaTableParams<Real>, 1, Dev> rama_table_params,
    TView<Real, 4, Dev> omega_tables,
    TView<RamaTableParams<Real>, 1, Dev> omega_table_params,

    TView<Int, 2, Dev> dispatch_indices,  // from forward pass
    TView<Real, 2, Dev> dTdV  // nterms x nposes x (1|len) x (1|len)
    ) -> TPack<Vec<Real, 3>, 2, Dev> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  
  int const max_n_conn = pose_stack_inter_block_connections.size(2);
  int const n_block_types = block_type_atom_downstream_of_conn.size(0);
  int const max_n_atoms_per_block_type = block_type_atom_downstream_of_conn.size(1);
  int const n_rama_tables = rama_tables.size(0);
  int const n_omega_tables = omega_tables.size(0);

  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);

  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);

  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);

  assert(block_type_rama_table.size(0) == n_block_types);
  assert(block_type_omega_table.size(0) == n_block_types);
  assert(block_type_lower_conn_ind.size(0) == n_block_types);
  assert(block_type_upper_conn_ind.size(0) == n_block_types);

  assert(block_type_torsion_atoms.size(0) == n_block_types);
  assert(block_type_torsion_atoms.size(1) == 12);

  assert(rama_table_params.size(0) == n_rama_tables);
  assert(omega_table_params.size(0) == n_omega_tables);

  auto dV_dxyz_t = TPack<Vec<Real, 3>, 2, Dev>::zeros({2, n_atoms});

  auto dV_dxyz = dV_dxyz_t.view;

  LAUNCH_BOX_32;
  // Define nt
  CTA_LAUNCH_T_PARAMS;

  auto rama_omega_func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_ind = dispatch_indices[0][ind];

    int const rot_ind1 = dispatch_indices[1][ind];
    int const rot_ind2 = dispatch_indices[2][ind];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    // Where did we write the output?
    // In block-pair-scoring mode, we store one energy per rotamer;
    // In non-block-pair-scoring mode, we store one energy per pose
    // (using atomic_add to accumulate for each block/rotamer in the pose)
    // NOTE: we only ever call this function if output_block_pair_energies is true
    // so previous logic just boils down to V_ind = ind;
    int const V_ind = ind;

    int const upper_nbr_is_pro = block_type_is_pro[block_type2];
    int const rama_table_ind =
        block_type_rama_table[block_type1][upper_nbr_is_pro];
    int const omega_table_ind =
        block_type_omega_table[block_type1][upper_nbr_is_pro];

    int const rot_offset1 = rot_coord_offset[rot_ind1];

    bool valid_phipsi = true;
    Vec<Int, 4> phi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i];
      phi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= phi_ats[i] != -1;
    }
    CoordQuad<Real> phi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        phi_coords.row(i) = rot_coords[phi_ats[i]];
      }
    }

    Vec<Int, 4> psi_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 4];
      psi_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);
      valid_phipsi &= psi_ats[i] != -1;
    }
    CoordQuad<Real> psi_coords;
    if (valid_phipsi) {
      for (int i = 0; i < 4; ++i) {
        psi_coords.row(i) = rot_coords[psi_ats[i]];
      }
    }

    // accumulate rama
    Real rama_block_weight = dTdV[0][V_ind];
    if (valid_phipsi && rama_table_ind >= 0) {
      auto rama = rama_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          rama_tables[rama_table_ind],
          Eigen::Map<Vec<Real, 2>>(rama_table_params[rama_table_ind].bbstarts),
          Eigen::Map<Vec<Real, 2>>(rama_table_params[rama_table_ind].bbsteps));
      for (int j = 0; j < 4; ++j) {
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][phi_ats[j]],
            common::get<1>(rama).row(j) * rama_block_weight);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[0][psi_ats[j]],
            common::get<2>(rama).row(j) * rama_block_weight);
      }
    }

    if (omega_table_ind < 0) {
      return;
    }

    CoordQuad<Real> omega_coords;
    Vec<Int, 4> omega_ats;
    for (int i = 0; i < 4; ++i) {
      UnresolvedAtomID<Int> i_at = block_type_torsion_atoms[block_type1][i + 8];

      // Check to see if the omega uaids are actually defined for this block
      // type. If the atom offset [0] or the connection index [1] are both -1,
      // this is a sentinel for an undefined omega.
      if (i_at.atom_id == -1 && i_at.conn_id == -1)
        return;  // no omega and we already calculated rama

      omega_ats[i] = resolve_rotamer_pair_atom_from_uaid(
          i_at,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type2,
          pose_ind,
          rot_coord_offset,
          first_rot_for_block,
          first_rot_block_type,
          pose_stack_inter_block_connections,
          block_type_atom_downstream_of_conn);

      // omega_atom_ind == -1 -> UAID resolution failed
      if (omega_ats[i] == -1)
        return;  // no omega and we already calculated rama
    }

    for (int i = 0; i < 4; ++i) {
      omega_coords.row(i) = rot_coords[omega_ats[i]];
    }

    Real omega_block_weight = dTdV[1][V_ind];
    if (valid_phipsi && omega_table_ind >= 0) {
      auto omega = omega_bbdep_V_dV<Dev, Real, Int>(
          phi_coords,
          psi_coords,
          omega_coords,
          omega_tables[omega_table_ind][0],
          omega_tables[omega_table_ind][1],
          Eigen::Map<Vec<Real, 2>>(
              omega_table_params[omega_table_ind].bbstarts),
          Eigen::Map<Vec<Real, 2>>(omega_table_params[omega_table_ind].bbsteps),
          32.8);
      for (int j = 0; j < 4; ++j) {
        // omega : [V, dVdphi, dVdpsi, dVdomega]
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][phi_ats[j]],
            common::get<1>(omega).row(j) * omega_block_weight);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][psi_ats[j]],
            common::get<2>(omega).row(j) * omega_block_weight);
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][omega_ats[j]],
            common::get<3>(omega).row(j) * omega_block_weight);
      }
    } else {
      // if rama is undefined, fall back to old version
      auto omega = omega_V_dV<Dev, Real, Int>(omega_coords, 32.8);
      for (int j = 0; j < 4; ++j) {
        // omega : [V, dVdomega]
        accumulate<Dev, Vec<Real, 3>>::add(
            dV_dxyz[1][omega_ats[j]],
            common::get<1>(omega).row(j) * omega_block_weight);
      }
    }
  });

  DeviceDispatch<Dev>::template forall<launch_t>(dispatch_indices.size(1), rama_omega_func);

  return dV_dxyz_t;
};

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
