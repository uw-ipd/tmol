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
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto DunbrackPoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
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
    TView<UnresolvedAtomID<Int>, 2, D> block_phi_uaids,
    TView<UnresolvedAtomID<Int>, 2, D> block_psi_uaids,
    TView<UnresolvedAtomID<Int>, 3, D>
        block_chi_uaids,  // TODO: unused, can probably be removed
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

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>> {
  int const n_poses = coords.size(0);
  int const n_block_types = block_n_dihedrals.size(0);
  int const max_n_atoms = coords.size(1);
  int const max_n_blocks = pose_stack_block_coord_offset.size(1);
  int const max_n_dih = block_dih_uaids.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);

  int const DIH_N_ATOMS = 4;  // TODO: is there a global for this?

  assert(coords.size(0) == n_poses);
  assert(coords.size(1) == max_n_atoms);

  assert(pose_stack_block_coord_offset.size(0) == n_poses);
  assert(pose_stack_block_coord_offset.size(1) == max_n_blocks);

  assert(pose_stack_block_type.size(0) == n_poses);
  assert(pose_stack_block_type.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
  assert(block_type_atom_downstream_of_conn.size(1) == max_n_conns);
  // assert(block_type_atom_downstream_of_conn.size(2) == ); TODO: what?

  assert(block_n_dihedrals.size(0) == n_block_types);

  assert(block_phi_uaids.size(0) == n_block_types);
  assert(block_phi_uaids.size(1) == DIH_N_ATOMS);

  assert(block_psi_uaids.size(0) == n_block_types);
  assert(block_psi_uaids.size(1) == DIH_N_ATOMS);

  assert(block_chi_uaids.size(0) == n_block_types);
  assert(block_chi_uaids.size(1) == DIH_N_ATOMS);

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

  auto V_t = TPack<Real, 2, D>::zeros({3, n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({3, n_poses, max_n_atoms});

  auto dihedral_atom_inds_t = TPack<Vec<Int, DIH_N_ATOMS>, 3, D>::zeros(
      {n_poses, max_n_blocks, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t =
      TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t =
      TPack<Eigen::Matrix<Real, DIH_N_ATOMS, 3>, 3, D>::zeros(
          {n_poses, max_n_blocks, max_n_dih});
  auto dihedral_deriv = dihedral_deriv_t.view;

  auto rotameric_rottable_assignment_t =
      TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto rotameric_rottable_assignment = rotameric_rottable_assignment_t.view;

  auto semirotameric_rottable_assignment_t =
      TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto semirotameric_rottable_assignment =
      semirotameric_rottable_assignment_t.view;

  auto dneglnprob_rot_dbb_xyz_t =
      TPack<CoordQuad, 3, D>::zeros({n_poses, max_n_blocks, 2});
  auto dneglnprob_rot_dbb_xyz = dneglnprob_rot_dbb_xyz_t.view;

  auto drotchi_devpen_dtor_xyz_t =
      TPack<CoordQuad, 3, D>::zeros({n_poses, max_n_blocks, 3});
  auto drotchi_devpen_dtor_xyz = drotchi_devpen_dtor_xyz_t.view;

  auto dneglnprob_nonrot_dtor_xyz_t =
      TPack<CoordQuad, 3, D>::zeros({n_poses, max_n_blocks, 3});
  auto dneglnprob_nonrot_dtor_xyz = dneglnprob_nonrot_dtor_xyz_t.view;

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(int pose_index, int block_index) {
    int block_type_index = pose_stack_block_type[pose_index][block_index];

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

        dihedral_atom_inds[pose_index][block_index][ii][jj] =
            resolve_atom_from_uaid(
                uaid,
                block_index,
                pose_index,
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[pose_index][block_index][ii][jj]
            == -1) {  // UAID resolution failed
          fail = true;
          break;
        }
      }
      if (fail) {  // if the dihedral resolution failed, let's fill the cached
                   // value with -1s since we might have partially filled it
                   // above
        dihedral_atom_inds[pose_index][block_index][ii] << -1, -1, -1, -1;
      }

      const Real PHI_DEFAULT = -60.0 * M_PI / 180;
      const Real PSI_DEFAULT = 60.0 * M_PI / 180;

      Real dih_default = (ii == 0)   ? PHI_DEFAULT
                         : (ii == 1) ? PSI_DEFAULT
                                     : 0.0;

      measure_dihedral_V_dV(
          coords[pose_index],
          dihedral_atom_inds[pose_index][block_index][ii],
          dih_default,
          dihedral_values[pose_index][block_index][ii],
          dihedral_deriv[pose_index][block_index][ii]);
    }

    // Templated on there being 2 backbone dihedrals for canonical aas.
    classify_rotamer_for_block<2>(
        dihedral_values[pose_index][block_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_rottable_assignment[pose_index][block_index],
        semirotameric_rottable_assignment[pose_index][block_index]);

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
          dihedral_values[pose_index][block_index],
          rotameric_rottable_assignment[pose_index][block_index],
          dneglnprob_rot_dbb_xyz[pose_index][block_index],
          dihedral_deriv[pose_index][block_index]);

      common::accumulate<D, Real>::add(V[0][pose_index], prob);

      Vec<Int, DIH_N_ATOMS> phi_ats =
          dihedral_atom_inds[pose_index][block_index][0];
      Vec<Int, DIH_N_ATOMS> psi_ats =
          dihedral_atom_inds[pose_index][block_index][1];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (phi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][phi_ats[j]],
              dneglnprob_rot_dbb_xyz[pose_index][block_index][0].row(j));
        if (psi_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[0][pose_index][psi_ats[j]],
              dneglnprob_rot_dbb_xyz[pose_index][block_index][1].row(j));
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
          dihedral_values[pose_index][block_index],
          ii,
          rotameric_rottable_assignment[pose_index][block_index],
          // Out
          drotchi_devpen_dtor_xyz[pose_index][block_index],
          dihedral_deriv[pose_index][block_index]);

      common::accumulate<D, Real>::add(V[1][pose_index], Erotdev);

      Vec<Int, DIH_N_ATOMS> tor0_ats =
          dihedral_atom_inds[pose_index][block_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats =
          dihedral_atom_inds[pose_index][block_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats =
          dihedral_atom_inds[pose_index][block_index][2 + ii];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][pose_index][tor0_ats[j]],
              drotchi_devpen_dtor_xyz[pose_index][block_index][0].row(j));
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][pose_index][tor1_ats[j]],
              drotchi_devpen_dtor_xyz[pose_index][block_index][1].row(j));
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[1][pose_index][tor2_ats[j]],
              drotchi_devpen_dtor_xyz[pose_index][block_index][2].row(j));
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

          dihedral_values[pose_index][block_index],
          semirotameric_rottable_assignment[pose_index][block_index],

          dneglnprob_nonrot_dtor_xyz[pose_index][block_index],
          dihedral_deriv[pose_index][block_index]);

      common::accumulate<D, Real>::add(V[2][pose_index], Esemi);

      int last = block_n_chi[block_type_index] + 1;
      Vec<Int, DIH_N_ATOMS> tor0_ats =
          dihedral_atom_inds[pose_index][block_index][0];
      Vec<Int, DIH_N_ATOMS> tor1_ats =
          dihedral_atom_inds[pose_index][block_index][1];
      Vec<Int, DIH_N_ATOMS> tor2_ats =
          dihedral_atom_inds[pose_index][block_index][last];
      for (int j = 0; j < DIH_N_ATOMS; ++j) {
        if (tor0_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][pose_index][tor0_ats[j]],
              dneglnprob_nonrot_dtor_xyz[pose_index][block_index][0].row(j));
        if (tor1_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][pose_index][tor1_ats[j]],
              dneglnprob_nonrot_dtor_xyz[pose_index][block_index][1].row(j));
        if (tor2_ats[j] != -1)
          accumulate<D, Vec<Real, 3>>::add(
              dV_dx[2][pose_index][tor2_ats[j]],
              dneglnprob_nonrot_dtor_xyz[pose_index][block_index][2].row(j));
      }
    }
  });

  DeviceDispatch<D>::forall_stacks(n_poses, max_n_blocks, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
