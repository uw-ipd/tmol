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
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/dunbrack/potentials/dunbrack_pose_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    bool output_block_pair_energies,

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);

  int n_V = output_block_pair_energies ? max_n_blocks : 1;
  TPack<Real, 4, D> V_t = TPack<Real, 4, D>::zeros({3, n_poses, n_V, n_V});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});

  auto dihedral_atom_inds_t =
      TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = TPack<Real, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t =
      TPack<Eigen::Matrix<Real, DIH_N_ATOMS, 3>, 2, D>::zeros(
          {n_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  auto rotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto rotameric_rottable_assignment = rotameric_rottable_assignment_t.view;

  auto semirotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto semirotameric_rottable_assignment =
      semirotameric_rottable_assignment_t.view;

  auto dneglnprob_rot_dbb_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 2});
  auto dneglnprob_rot_dbb_xyz = dneglnprob_rot_dbb_xyz_t.view;

  auto drotchi_devpen_dtor_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto drotchi_devpen_dtor_xyz = drotchi_devpen_dtor_xyz_t.view;

  auto dneglnprob_nonrot_dtor_xyz_t =
      TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto dneglnprob_nonrot_dtor_xyz = dneglnprob_nonrot_dtor_xyz_t.view;

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_index = ind / max_n_blocks;
    int const block_index = ind % max_n_blocks;
    int const block_type_index = first_rot_block_type[pose_index][block_index];
    if (block_type_index == -1) {
      // in non-rotamer scoring (both whole-pose and block-pair scoring)
      // we will launch n_poses * max_n_blocks_per_pose threads
      // with some threads being launched for blocks with -1 block type.
      return;
    }

    if (block_rotamer_table_set[block_type_index] == -1) return;
    int const rotamer_index = first_rot_for_block[pose_index][block_index];
    if (rotamer_index == -1) {
      // This should never happen
      return;
    }
    printf(
        "Dunbrack forward p %d b %d r %d\n",
        pose_index,
        block_index,
        rotamer_index);

    for (int ii = 0; ii < block_n_dihedrals[block_type_index]; ii++) {
      auto dih_uaids = block_dih_uaids[block_type_index][ii];
      bool fail = false;
      for (int jj = 0; jj < DIH_N_ATOMS; jj++) {
        UnresolvedAtomID<Int> uaid = dih_uaids[jj];

        if (uaid.atom_id == -1 && uaid.conn_id == -1) {  // Dihedral undefined
          fail = true;
          break;
        }

        dihedral_atom_inds[rotamer_index][ii][jj] =
            resolve_rotamer_atom_from_uaid(
                uaid,
                rotamer_index,
                block_index,
                pose_index,
                rot_coord_offset,
                first_rot_for_block,
                first_rot_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[rotamer_index][ii][jj] == -1) {
          // UAID resolution failed
          fail = true;
          break;
        }
      }
      if (fail) {  // if the dihedral resolution failed, let's fill the cached
                   // value with -1s since we might have partially filled it
                   // above
        dihedral_atom_inds[rotamer_index][ii] << -1, -1, -1, -1;
      }

      const Real PHI_DEFAULT = -60.0 * M_PI / 180;
      const Real PSI_DEFAULT = 60.0 * M_PI / 180;

      Real dih_default = (ii == 0)   ? PHI_DEFAULT
                         : (ii == 1) ? PSI_DEFAULT
                                     : 0.0;

      measure_dihedral_V_dV(
          TensorAccessor<Vec<Real, 3>, 1, D>(rot_coords),
          dihedral_atom_inds[rotamer_index][ii],
          dih_default,
          dihedral_values[rotamer_index][ii],
          dihedral_deriv[rotamer_index][ii]);
    }

    // Templated on there being 2 backbone dihedrals for canonical aas.
    classify_rotamer_for_block<2>(
        dihedral_values[rotamer_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_rottable_assignment[rotamer_index],
        semirotameric_rottable_assignment[rotamer_index]);

    if (block_rotameric_index[block_type_index] != -1) {
      Real prob = rotameric_chi_probability_for_block(
          rotameric_neglnprob_tables,
          rotprob_table_sizes,
          rotprob_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          block_probability_table_offset[block_type_index],
          block_rotamer_table_set[block_type_index],
          dihedral_values[rotamer_index],
          rotameric_rottable_assignment[rotamer_index],
          dneglnprob_rot_dbb_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      if (output_block_pair_energies) {
        V[0][pose_index][block_index][block_index] = prob;
      } else {
        common::accumulate<D, Real>::add(V[0][pose_index][0][0], prob);
      }

      // Note that we will accumulate all of the dV_dx derivatives
      // into the phi and psi definiing atoms of the _first rotamers_
      // of residues i+1 and i-1 respectively. This is dedicedly weird
      // unless there is only one rotmaer for each residue
      Vec<Int, DIH_N_ATOMS> phi_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> psi_ats = dihedral_atom_inds[rotamer_index][1];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (phi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][phi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][0].row(j));
        if (psi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][psi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][1].row(j));
      }
    }

    for (int ii = 0; ii < block_n_rotameric_chi[block_type_index]; ii++) {
      // deviation from chi for chis within this block
      auto Erotdev = block_deviation_penalty_for_chi(
          // DB tables
          rotameric_mean_tables,
          rotameric_sdev_tables,
          rotmean_table_sizes,
          rotmean_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          // Block type info
          block_rotamer_table_set[block_type_index],
          block_mean_table_offset[block_type_index],
          block_n_chi[block_type_index],
          // Block info
          dihedral_values[rotamer_index],
          ii,
          rotameric_rottable_assignment[rotamer_index],
          // Out
          drotchi_devpen_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      // common::accumulate<D, Real>::add(V[1][V_index], Erotdev);
      if (output_block_pair_energies) {
        V[1][pose_index][block_index][block_index] = Erotdev;
      } else {
        common::accumulate<D, Real>::add(V[1][pose_index][0][0], Erotdev);
      }

      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats =
          dihedral_atom_inds[rotamer_index][2 + ii];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor0_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][0].row(j));
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor1_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][1].row(j));
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor2_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][2].row(j));
      }
    }

    if (block_semirotameric_index[block_type_index] != -1) {
      auto Esemi = block_semirotameric_energy(
          semirotameric_tables,
          semirot_table_sizes,
          semirot_table_strides,
          semirot_start,
          semirot_step,
          semirot_periodicity,

          block_semirotameric_index[block_type_index],
          block_semirotameric_tableset_offset[block_type_index],
          block_n_chi[block_type_index] + 1,

          dihedral_values[rotamer_index],
          semirotameric_rottable_assignment[rotamer_index],

          dneglnprob_nonrot_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      // common::accumulate<D, Real>::add(V[2][V_index], Esemi);
      if (output_block_pair_energies) {
        V[2][pose_index][block_index][block_index] = Esemi;
      } else {
        common::accumulate<D, Real>::add(V[2][pose_index][0][0], Esemi);
      }

      int last = block_n_chi[block_type_index] + 1;  // = +2 - 1
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats = dihedral_atom_inds[rotamer_index][last];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor0_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][0].row(j));
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor1_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][1].row(j));
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor2_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][2].row(j));
      }
    }
  });

  DeviceDispatch<D>::template forall<launch_t>(n_poses * max_n_blocks, func);
  //   DeviceDispatch<D>::synchronize_device();

  return {V_t, dV_dx_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,

    TView<Real, 4, D> dTdV  // n_terms x n_poses x max_n_blocks x max_n_blocks
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});

  auto dihedral_atom_inds_t =
      TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = TPack<Real, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t =
      TPack<Eigen::Matrix<Real, DIH_N_ATOMS, 3>, 2, D>::zeros(
          {n_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  auto rotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto rotameric_rottable_assignment = rotameric_rottable_assignment_t.view;

  auto semirotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto semirotameric_rottable_assignment =
      semirotameric_rottable_assignment_t.view;

  auto dneglnprob_rot_dbb_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 2});
  auto dneglnprob_rot_dbb_xyz = dneglnprob_rot_dbb_xyz_t.view;

  auto drotchi_devpen_dtor_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto drotchi_devpen_dtor_xyz = drotchi_devpen_dtor_xyz_t.view;

  auto dneglnprob_nonrot_dtor_xyz_t =
      TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto dneglnprob_nonrot_dtor_xyz = dneglnprob_nonrot_dtor_xyz_t.view;

  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose_index = ind / max_n_blocks;
    int const block_index = ind % max_n_blocks;
    int const block_type_index = first_rot_block_type[pose_index][block_index];
    if (block_type_index == -1) {
      // in non-rotamer scoring (both whole-pose and block-pair scoring)
      // we will launch n_poses * max_n_blocks_per_pose threads
      // with some threads being launched for blocks with -1 block type.
      return;
    }

    if (block_rotamer_table_set[block_type_index] == -1) return;
    int const rotamer_index = first_rot_for_block[pose_index][block_index];
    if (rotamer_index == -1) {
      // This should never happen
      return;
    }
    printf(
        "Dunbrack backward p %d b %d r %d\n",
        pose_index,
        block_index,
        rotamer_index);

    for (int ii = 0; ii < block_n_dihedrals[block_type_index]; ii++) {
      auto dih_uaids = block_dih_uaids[block_type_index][ii];
      bool fail = false;
      for (int jj = 0; jj < DIH_N_ATOMS; jj++) {
        UnresolvedAtomID<Int> uaid = dih_uaids[jj];

        if (uaid.atom_id == -1 && uaid.conn_id == -1) {  // Dihedral undefined
          fail = true;
          break;
        }

        dihedral_atom_inds[rotamer_index][ii][jj] =
            resolve_rotamer_atom_from_uaid(
                uaid,
                rotamer_index,
                block_index,
                pose_index,
                rot_coord_offset,
                first_rot_for_block,
                first_rot_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[rotamer_index][ii][jj] == -1) {
          // UAID resolution failed
          fail = true;
          break;
        }
      }
      if (fail) {  // if the dihedral resolution failed, let's fill the cached
                   // value with -1s since we might have partially filled it
                   // above
        dihedral_atom_inds[rotamer_index][ii] << -1, -1, -1, -1;
      }

      const Real PHI_DEFAULT = -60.0 * M_PI / 180;
      const Real PSI_DEFAULT = 60.0 * M_PI / 180;

      Real dih_default = (ii == 0)   ? PHI_DEFAULT
                         : (ii == 1) ? PSI_DEFAULT
                                     : 0.0;

      measure_dihedral_V_dV(
          TensorAccessor<Vec<Real, 3>, 1, D>(rot_coords),
          dihedral_atom_inds[rotamer_index][ii],
          dih_default,
          dihedral_values[rotamer_index][ii],
          dihedral_deriv[rotamer_index][ii]);
    }

    // Templated on there being 2 backbone dihedrals for canonical aas.
    classify_rotamer_for_block<2>(
        dihedral_values[rotamer_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_rottable_assignment[rotamer_index],
        semirotameric_rottable_assignment[rotamer_index]);

    if (block_rotameric_index[block_type_index] != -1) {
      Real prob = rotameric_chi_probability_for_block(
          rotameric_neglnprob_tables,
          rotprob_table_sizes,
          rotprob_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          block_probability_table_offset[block_type_index],
          block_rotamer_table_set[block_type_index],
          dihedral_values[rotamer_index],
          rotameric_rottable_assignment[rotamer_index],
          dneglnprob_rot_dbb_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Vec<Int, DIH_N_ATOMS> phi_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> psi_ats = dihedral_atom_inds[rotamer_index][1];
      Real block_weight_0 = dTdV[0][pose_index][block_index][block_index];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (phi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][phi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][0].row(j) * block_weight_0);
        if (psi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][psi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][1].row(j) * block_weight_0);
      }
    }

    for (int ii = 0; ii < block_n_rotameric_chi[block_type_index]; ii++) {
      // deviation from chi for chis within this block
      auto Erotdev = block_deviation_penalty_for_chi(
          // DB tables
          rotameric_mean_tables,
          rotameric_sdev_tables,
          rotmean_table_sizes,
          rotmean_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          // Block type info
          block_rotamer_table_set[block_type_index],
          block_mean_table_offset[block_type_index],
          block_n_chi[block_type_index],
          // Block info
          dihedral_values[rotamer_index],
          ii,
          rotameric_rottable_assignment[rotamer_index],
          // Out
          drotchi_devpen_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Real block_weight_1 = dTdV[1][pose_index][block_index][block_index];
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats =
          dihedral_atom_inds[rotamer_index][2 + ii];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor0_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][0].row(j)
                  * block_weight_1);
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor1_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][1].row(j)
                  * block_weight_1);
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor2_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][2].row(j)
                  * block_weight_1);
      }
    }

    if (block_semirotameric_index[block_type_index] != -1) {
      auto Esemi = block_semirotameric_energy(
          semirotameric_tables,
          semirot_table_sizes,
          semirot_table_strides,
          semirot_start,
          semirot_step,
          semirot_periodicity,

          block_semirotameric_index[block_type_index],
          block_semirotameric_tableset_offset[block_type_index],
          block_n_chi[block_type_index] + 1,

          dihedral_values[rotamer_index],
          semirotameric_rottable_assignment[rotamer_index],

          dneglnprob_nonrot_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Real block_weight_2 = dTdV[2][pose_index][block_index][block_index];

      int last = block_n_chi[block_type_index] + 1;
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats = dihedral_atom_inds[rotamer_index][last];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor0_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][0].row(j)
                  * block_weight_2);
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor1_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][1].row(j)
                  * block_weight_2);
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor2_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][2].row(j)
                  * block_weight_2);
      }
    }
  });

  DeviceDispatch<D>::template forall<launch_t>(n_poses * max_n_blocks, func);

  return dV_dx_t;
}  // namespace potentials

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,
    bool output_block_pair_energies,

    bool compute_derivs

    ) -> std::
    tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);

  TPack<Real, 2, D> V_t;
  TPack<Int, 2, D> dispatch_indices_t;
  if (output_block_pair_energies) {
    V_t = TPack<Real, 2, D>::zeros({3, n_rots});
    dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_rots});
  } else {
    V_t = TPack<Real, 2, D>::zeros({3, n_poses});
    dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_poses});
  }
  // Derivative calculation on the forward pass for rotamers
  // will be inaccurate, but go ahead and do it anyways?
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});

  auto dihedral_atom_inds_t =
      TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = TPack<Real, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t =
      TPack<Eigen::Matrix<Real, DIH_N_ATOMS, 3>, 2, D>::zeros(
          {n_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  auto rotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto rotameric_rottable_assignment = rotameric_rottable_assignment_t.view;

  auto semirotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto semirotameric_rottable_assignment =
      semirotameric_rottable_assignment_t.view;

  auto dneglnprob_rot_dbb_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 2});
  auto dneglnprob_rot_dbb_xyz = dneglnprob_rot_dbb_xyz_t.view;

  auto drotchi_devpen_dtor_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto drotchi_devpen_dtor_xyz = drotchi_devpen_dtor_xyz_t.view;

  auto dneglnprob_nonrot_dtor_xyz_t =
      TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto dneglnprob_nonrot_dtor_xyz = dneglnprob_nonrot_dtor_xyz_t.view;

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto func = ([=] TMOL_DEVICE_FUNC(int rotamer_index) {
    int const pose_index = pose_ind_for_rot[rotamer_index];
    int const block_index = block_ind_for_rot[rotamer_index];
    int const block_type_index = block_type_ind_for_rot[rotamer_index];
    if (block_type_index == -1) {
      // in non-rotamer scoring (both whole-pose and block-pair scoring)
      // we will launch n_poses * max_n_blocks_per_pose threads
      // with some threads being launched for blocks with -1 block type.
      return;
    }

    // Where will we write the output?
    // In block-pair-scoring mode, we store one energy per rotamer;
    // In non-block-pair-scoring mode, we store one energy per pose
    // (using atomic_add to accumulate for each block/rotamer in the pose)
    int const V_index =
        (output_block_pair_energies) ? rotamer_index : pose_index;

    if (output_block_pair_energies) {
      // Label all rotamer pair indices -- unlabeled
      // entries in the dispatch_indices tensor will be
      // 0's, representing pose 0, rotamer 0, and their
      // corresponding entry in the V tensor of 0 will
      // get accumulated into this rotamer's one-body energy
      // without changing it.
      dispatch_indices[0][rotamer_index] = pose_index;
      dispatch_indices[1][rotamer_index] = rotamer_index;
      dispatch_indices[2][rotamer_index] = rotamer_index;
    }

    if (block_rotamer_table_set[block_type_index] == -1) return;

    for (int ii = 0; ii < block_n_dihedrals[block_type_index]; ii++) {
      auto dih_uaids = block_dih_uaids[block_type_index][ii];
      bool fail = false;
      for (int jj = 0; jj < DIH_N_ATOMS; jj++) {
        UnresolvedAtomID<Int> uaid = dih_uaids[jj];

        if (uaid.atom_id == -1 && uaid.conn_id == -1) {  // Dihedral undefined
          fail = true;
          break;
        }

        dihedral_atom_inds[rotamer_index][ii][jj] =
            resolve_rotamer_atom_from_uaid(
                uaid,
                rotamer_index,
                block_index,
                pose_index,
                rot_coord_offset,
                first_rot_for_block,
                first_rot_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[rotamer_index][ii][jj] == -1) {
          // UAID resolution failed
          fail = true;
          break;
        }
      }
      if (fail) {  // if the dihedral resolution failed, let's fill the cached
                   // value with -1s since we might have partially filled it
                   // above
        dihedral_atom_inds[rotamer_index][ii] << -1, -1, -1, -1;
      }

      const Real PHI_DEFAULT = -60.0 * M_PI / 180;
      const Real PSI_DEFAULT = 60.0 * M_PI / 180;

      Real dih_default = (ii == 0)   ? PHI_DEFAULT
                         : (ii == 1) ? PSI_DEFAULT
                                     : 0.0;

      measure_dihedral_V_dV(
          TensorAccessor<Vec<Real, 3>, 1, D>(rot_coords),
          dihedral_atom_inds[rotamer_index][ii],
          dih_default,
          dihedral_values[rotamer_index][ii],
          dihedral_deriv[rotamer_index][ii]);
    }

    // Templated on there being 2 backbone dihedrals for canonical aas.
    classify_rotamer_for_block<2>(
        dihedral_values[rotamer_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_rottable_assignment[rotamer_index],
        semirotameric_rottable_assignment[rotamer_index]);

    if (block_rotameric_index[block_type_index] != -1) {
      Real prob = rotameric_chi_probability_for_block(
          rotameric_neglnprob_tables,
          rotprob_table_sizes,
          rotprob_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          block_probability_table_offset[block_type_index],
          block_rotamer_table_set[block_type_index],
          dihedral_values[rotamer_index],
          rotameric_rottable_assignment[rotamer_index],
          dneglnprob_rot_dbb_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      common::accumulate<D, Real>::add(V[0][V_index], prob);

      // Note that we will accumulate all of the dV_dx derivatives
      // into the phi and psi definiing atoms of the _first rotamers_
      // of residues i+1 and i-1 respectively. This is dedicedly weird
      // unless there is only one rotmaer for each residue
      Vec<Int, DIH_N_ATOMS> phi_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> psi_ats = dihedral_atom_inds[rotamer_index][1];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (phi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][phi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][0].row(j));
        if (psi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][psi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][1].row(j));
      }
    }

    for (int ii = 0; ii < block_n_rotameric_chi[block_type_index]; ii++) {
      // deviation from chi for chis within this block
      auto Erotdev = block_deviation_penalty_for_chi(
          // DB tables
          rotameric_mean_tables,
          rotameric_sdev_tables,
          rotmean_table_sizes,
          rotmean_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          // Block type info
          block_rotamer_table_set[block_type_index],
          block_mean_table_offset[block_type_index],
          block_n_chi[block_type_index],
          // Block info
          dihedral_values[rotamer_index],
          ii,
          rotameric_rottable_assignment[rotamer_index],
          // Out
          drotchi_devpen_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      common::accumulate<D, Real>::add(V[1][V_index], Erotdev);

      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats =
          dihedral_atom_inds[rotamer_index][2 + ii];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor0_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][0].row(j));
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor1_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][1].row(j));
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor2_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][2].row(j));
      }
    }

    if (block_semirotameric_index[block_type_index] != -1) {
      auto Esemi = block_semirotameric_energy(
          semirotameric_tables,
          semirot_table_sizes,
          semirot_table_strides,
          semirot_start,
          semirot_step,
          semirot_periodicity,

          block_semirotameric_index[block_type_index],
          block_semirotameric_tableset_offset[block_type_index],
          block_n_chi[block_type_index] + 1,

          dihedral_values[rotamer_index],
          semirotameric_rottable_assignment[rotamer_index],

          dneglnprob_nonrot_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      common::accumulate<D, Real>::add(V[2][V_index], Esemi);

      int last = block_n_chi[block_type_index] + 1;  // = +2 - 1
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats = dihedral_atom_inds[rotamer_index][last];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor0_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][0].row(j));
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor1_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][1].row(j));
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor2_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][2].row(j));
      }
    }
  });

  DeviceDispatch<D>::template forall<launch_t>(n_rots, func);
  //   DeviceDispatch<D>::synchronize_device();

  return {V_t, dV_dx_t, dispatch_indices_t};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    // common params
    TView<Vec<Real, 3>, 1, D> rot_coords,
    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 1, D> pose_ind_for_atom,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Int, 1, D> block_ind_for_rot,
    TView<Int, 1, D> pose_ind_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    Int max_n_rots_per_pose,

    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn,

    TView<Real, 3, D> rotameric_neglnprob_tables,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotprob_table_strides,
    TView<Real, 3, D> rotameric_mean_tables,
    TView<Real, 3, D> rotameric_sdev_tables,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_sizes,
    TView<Vec<int64_t, 2>, 1, D> rotmean_table_strides,

    TView<Vec<Real, 2>, 1, D> rotameric_bb_start,        // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_step,         // ntable-set entries
    TView<Vec<Real, 2>, 1, D> rotameric_bb_periodicity,  // ntable-set entries

    TView<Int, 1, D> rotameric_rotind2tableind,
    TView<Int, 1, D> semirotameric_rotind2tableind,

    TView<Real, 4, D> semirotameric_tables,              // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_sizes,    // n-semirot-tabset
    TView<Vec<int64_t, 3>, 1, D> semirot_table_strides,  // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_start,             // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_step,              // n-semirot-tabset
    TView<Vec<Real, 3>, 1, D> semirot_periodicity,       // n-semirot-tabset

    TView<Int, 1, D> block_n_dihedrals,
    TView<UnresolvedAtomID<Int>, 3, D> block_dih_uaids,
    TView<Int, 1, D> block_rotamer_table_set,
    TView<Int, 1, D> block_rotameric_index,
    TView<Int, 1, D> block_semirotameric_index,
    TView<Int, 1, D> block_n_chi,
    TView<Int, 1, D> block_n_rotameric_chi,
    TView<Int, 1, D> block_probability_table_offset,
    TView<Int, 1, D> block_mean_table_offset,
    TView<Int, 1, D> block_rotamer_index_to_table_index,
    TView<Int, 1, D> block_semirotameric_tableset_offset,

    TView<Real, 2, D> dTdV  // n_terms x n_rotamers
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_n_dihedrals.size(0);

  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;

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

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  //     block_type_atom_downstream_of_conn.size(2) == max number of atoms in
  //     any block type

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_dih_uaids.size(0) == n_block_types);
  assert(block_dih_uaids.size(1) == max_n_dih);
  assert(block_dih_uaids.size(2) == DIH_N_ATOMS);

  assert(block_rotamer_table_set.size(0) == n_block_types);
  assert(block_rotameric_index.size(0) == n_block_types);
  assert(block_semirotameric_index.size(0) == n_block_types);
  assert(block_n_chi.size(0) == n_block_types);
  assert(block_n_rotameric_chi.size(0) == n_block_types);
  assert(block_probability_table_offset.size(0) == n_block_types);
  assert(block_mean_table_offset.size(0) == n_block_types);
  assert(block_rotamer_index_to_table_index.size(0) == n_block_types);
  assert(block_semirotameric_tableset_offset.size(0) == n_block_types);

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});

  auto dihedral_atom_inds_t =
      TPack<Vec<Int, DIH_N_ATOMS>, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t = TPack<Real, 2, D>::zeros({n_rots, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t =
      TPack<Eigen::Matrix<Real, DIH_N_ATOMS, 3>, 2, D>::zeros(
          {n_rots, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  auto rotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto rotameric_rottable_assignment = rotameric_rottable_assignment_t.view;

  auto semirotameric_rottable_assignment_t = TPack<Int, 1, D>::zeros({n_rots});
  auto semirotameric_rottable_assignment =
      semirotameric_rottable_assignment_t.view;

  auto dneglnprob_rot_dbb_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 2});
  auto dneglnprob_rot_dbb_xyz = dneglnprob_rot_dbb_xyz_t.view;

  auto drotchi_devpen_dtor_xyz_t = TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto drotchi_devpen_dtor_xyz = drotchi_devpen_dtor_xyz_t.view;

  auto dneglnprob_nonrot_dtor_xyz_t =
      TPack<CoordQuad, 2, D>::zeros({n_rots, 3});
  auto dneglnprob_nonrot_dtor_xyz = dneglnprob_nonrot_dtor_xyz_t.view;

  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int rotamer_index) {
    int const pose_index = pose_ind_for_rot[rotamer_index];
    int const block_index = block_ind_for_rot[rotamer_index];
    int const block_type_index = block_type_ind_for_rot[rotamer_index];
    if (block_type_index == -1) {
      // in non-rotamer scoring (both whole-pose and block-pair scoring)
      // we will launch n_poses * max_n_blocks_per_pose threads
      // with some threads being launched for blocks with -1 block type.
      return;
    }

    if (block_rotamer_table_set[block_type_index] == -1) return;

    for (int ii = 0; ii < block_n_dihedrals[block_type_index]; ii++) {
      auto dih_uaids = block_dih_uaids[block_type_index][ii];
      bool fail = false;
      for (int jj = 0; jj < DIH_N_ATOMS; jj++) {
        UnresolvedAtomID<Int> uaid = dih_uaids[jj];

        if (uaid.atom_id == -1 && uaid.conn_id == -1) {  // Dihedral undefined
          fail = true;
          break;
        }

        dihedral_atom_inds[rotamer_index][ii][jj] =
            resolve_rotamer_atom_from_uaid(
                uaid,
                rotamer_index,
                block_index,
                pose_index,
                rot_coord_offset,
                first_rot_for_block,
                first_rot_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[rotamer_index][ii][jj] == -1) {
          // UAID resolution failed
          fail = true;
          break;
        }
      }
      if (fail) {  // if the dihedral resolution failed, let's fill the cached
                   // value with -1s since we might have partially filled it
                   // above
        dihedral_atom_inds[rotamer_index][ii] << -1, -1, -1, -1;
      }

      const Real PHI_DEFAULT = -60.0 * M_PI / 180;
      const Real PSI_DEFAULT = 60.0 * M_PI / 180;

      Real dih_default = (ii == 0)   ? PHI_DEFAULT
                         : (ii == 1) ? PSI_DEFAULT
                                     : 0.0;

      measure_dihedral_V_dV(
          TensorAccessor<Vec<Real, 3>, 1, D>(rot_coords),
          dihedral_atom_inds[rotamer_index][ii],
          dih_default,
          dihedral_values[rotamer_index][ii],
          dihedral_deriv[rotamer_index][ii]);
    }

    // Templated on there being 2 backbone dihedrals for canonical aas.
    classify_rotamer_for_block<2>(
        dihedral_values[rotamer_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_rottable_assignment[rotamer_index],
        semirotameric_rottable_assignment[rotamer_index]);

    if (block_rotameric_index[block_type_index] != -1) {
      Real prob = rotameric_chi_probability_for_block(
          rotameric_neglnprob_tables,
          rotprob_table_sizes,
          rotprob_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          block_probability_table_offset[block_type_index],
          block_rotamer_table_set[block_type_index],
          dihedral_values[rotamer_index],
          rotameric_rottable_assignment[rotamer_index],
          dneglnprob_rot_dbb_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Vec<Int, DIH_N_ATOMS> phi_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> psi_ats = dihedral_atom_inds[rotamer_index][1];
      Real block_weight_0 = dTdV[0][rotamer_index];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (phi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][phi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][0].row(j) * block_weight_0);
        if (psi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][psi_ats[j]],
              dneglnprob_rot_dbb_xyz[rotamer_index][1].row(j) * block_weight_0);
      }
    }

    for (int ii = 0; ii < block_n_rotameric_chi[block_type_index]; ii++) {
      // deviation from chi for chis within this block
      auto Erotdev = block_deviation_penalty_for_chi(
          // DB tables
          rotameric_mean_tables,
          rotameric_sdev_tables,
          rotmean_table_sizes,
          rotmean_table_strides,
          rotameric_bb_start,
          rotameric_bb_step,
          rotameric_bb_periodicity,
          // Block type info
          block_rotamer_table_set[block_type_index],
          block_mean_table_offset[block_type_index],
          block_n_chi[block_type_index],
          // Block info
          dihedral_values[rotamer_index],
          ii,
          rotameric_rottable_assignment[rotamer_index],
          // Out
          drotchi_devpen_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Real block_weight_1 = dTdV[1][rotamer_index];
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats =
          dihedral_atom_inds[rotamer_index][2 + ii];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor0_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][0].row(j)
                  * block_weight_1);
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor1_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][1].row(j)
                  * block_weight_1);
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][tor2_ats[j]],
              drotchi_devpen_dtor_xyz[rotamer_index][2].row(j)
                  * block_weight_1);
      }
    }

    if (block_semirotameric_index[block_type_index] != -1) {
      auto Esemi = block_semirotameric_energy(
          semirotameric_tables,
          semirot_table_sizes,
          semirot_table_strides,
          semirot_start,
          semirot_step,
          semirot_periodicity,

          block_semirotameric_index[block_type_index],
          block_semirotameric_tableset_offset[block_type_index],
          block_n_chi[block_type_index] + 1,

          dihedral_values[rotamer_index],
          semirotameric_rottable_assignment[rotamer_index],

          dneglnprob_nonrot_dtor_xyz[rotamer_index],
          dihedral_deriv[rotamer_index]);

      Real block_weight_2 = dTdV[2][rotamer_index];

      int last = block_n_chi[block_type_index] + 1;
      Vec<Int, DIH_N_ATOMS> tor0_ats = dihedral_atom_inds[rotamer_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats = dihedral_atom_inds[rotamer_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats = dihedral_atom_inds[rotamer_index][last];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor0_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][0].row(j)
                  * block_weight_2);
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor1_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][1].row(j)
                  * block_weight_2);
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][tor2_ats[j]],
              dneglnprob_nonrot_dtor_xyz[rotamer_index][2].row(j)
                  * block_weight_2);
      }
    }
  });

  DeviceDispatch<D>::template forall<launch_t>(n_rots, func);

  return dV_dx_t;
}  // namespace potentials

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
