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

    // TView<DunbrackGlobalParams<Real>, 1, D> global_params,
    // TView<DunbrackGlobalParams<Real>, 1, D> global_params,
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

    TView<Int, 1, D> block_n_dihredrals,
    TView<UnresolvedAtomID<Int>, 2, D> block_phi_uaids,
    TView<UnresolvedAtomID<Int>, 2, D> block_psi_uaids,
    TView<UnresolvedAtomID<Int>, 3, D> block_chi_uaids,
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
  int const max_n_atoms = coords.size(1);
  int const max_n_blocks = pose_stack_block_coord_offset.size(1);
  int const max_n_dih = block_dih_uaids.size(1);
  auto V_t = TPack<Real, 2, D>::zeros({3, n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({3, n_poses, max_n_atoms});

  auto dihedral_atom_inds_t =
      TPack<Vec<Int, 4>, 3, D>::zeros({n_poses, max_n_blocks, max_n_dih});
  auto dihedral_atom_inds = dihedral_atom_inds_t.view;
  auto dihedral_values_t =
      TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, max_n_dih});
  auto dihedral_values = dihedral_values_t.view;
  auto dihedral_deriv_t = TPack<Eigen::Matrix<Real, 4, 3>, 3, D>::zeros(
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

    for (int ii = 0; ii < block_n_dihredrals[block_type_index]; ii++) {
      auto dih_uaids = block_dih_uaids[block_type_index][ii];
      bool fail = false;
      for (int jj = 0; jj < 4; jj++) {
        UnresolvedAtomID<Int> uaid = dih_uaids[jj];

        if (uaid.atom_id == -1 && uaid.conn_id == -1) fail = true;

        dihedral_atom_inds[pose_index][block_index][ii][jj] =
            resolve_atom_from_uaid(
                uaid,
                block_index,
                pose_index,
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
        if (dihedral_atom_inds[pose_index][block_index][ii][jj] == -1) {
          // The UAID resolution failed! In this case, we should just skip this
          // block
          fail = true;
        }

        // measure_dihedral_V_dV()
      }
      if (fail) continue;

      measure_dihedral_V_dV(
          coords[pose_index],
          dihedral_atom_inds[pose_index][block_index][ii],
          dihedral_values[pose_index][block_index][ii],
          dihedral_deriv[pose_index][block_index][ii]);

      /*printf("RES%i: %i %i %i %i - %f\n",
        block_index,
        dih_atom_inds[0],
        dih_atom_inds[1],
        dih_atom_inds[2],
        dih_atom_inds[3],
        dihedral_values[pose_index][block_index][ii]);*/
    }

    // Templated on there being 2 backbone dihedrals for canonical aas.
    // if (nrotameric_chi_for_res[stack][i] >= 0) { // TODO: can't we just skip
    // the whole thing at the start? probably... printf("RES%i ", block_index);
    classify_rotamer_for_block<2>(
        dihedral_values[pose_index][block_index],
        block_n_rotameric_chi[block_type_index],
        block_rotamer_index_to_table_index[block_type_index],
        rotameric_rotind2tableind,
        semirotameric_rotind2tableind,
        rotameric_rottable_assignment[pose_index][block_index],
        semirotameric_rottable_assignment[pose_index][block_index]);

    // printf("%i Rotameric table ind: %i\n", block_index,
    // rotameric_rottable_assignment[pose_index][block_index]); printf("%i
    // Semirotameric table ind: %i\n", block_index,
    // semirotameric_rottable_assignment[pose_index][block_index]);

    if (block_rotameric_index[block_type_index] != -1) {
      // if(rotameric_rottable_assignment[pose_index][block_index]) {
      // printf("%i calculating rotameric\n", block_index);
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
      // printf("%i - %f\n", block_index, prob);
      /*common::accumulate<D, Real>::add(V[0][pose_index], prob);

      Vec<Int, 4> phi_ats = dihedral_atom_inds[pose_index][block_index][0];
      Vec<Int, 4> psi_ats = dihedral_atom_inds[pose_index][block_index][1];
      for (int j = 0; j < 4; ++j) {
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[0][pose_index][phi_ats[j]],
      dneglnprob_rot_dbb_xyz[pose_index][block_index][0].row(j)); accumulate<D,
      Vec<Real, 3>>::add( dV_dx[0][pose_index][psi_ats[j]],
      dneglnprob_rot_dbb_xyz[pose_index][block_index][1].row(j));
      }*/
      // printf("%i rotameric_chi_prob: %f\n", block_index, prob);
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
      // printf("%i/%i deviation penalty: %f\n", block_index, ii, Erotdev);
      Vec<Int, 4> tor0_ats = dihedral_atom_inds[pose_index][block_index][0];
      Vec<Int, 4> tor1_ats = dihedral_atom_inds[pose_index][block_index][1];
      Vec<Int, 4> tor2_ats =
          dihedral_atom_inds[pose_index][block_index][2 + ii];
      for (int j = 0; j < 4; ++j) {
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[1][pose_index][tor0_ats[j]],
            drotchi_devpen_dtor_xyz[pose_index][block_index][0].row(j));
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[1][pose_index][tor1_ats[j]],
            drotchi_devpen_dtor_xyz[pose_index][block_index][1].row(j));
        accumulate<D, Vec<Real, 3>>::add(
            dV_dx[1][pose_index][tor2_ats[j]],
            drotchi_devpen_dtor_xyz[pose_index][block_index][2].row(j));
      }
    }

    if (block_semirotameric_index[block_type_index] != -1) {
      /*printf("%i calculating semirot energy\n", block_index);
      printf(
          "%i semiindex:%i semioffset:%i nrotchi:%i assign:%i\n",
          block_index,
          block_semirotameric_index[block_type_index],
          block_semirotameric_tableset_offset[block_type_index],
          block_n_rotameric_chi[block_type_index],
          semirotameric_rottable_assignment[pose_index][block_index]);*/
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
      /*common::accumulate<D, Real>::add(V[2][pose_index], Esemi);
      //printf("%i semirot: %f\n", block_index, Esemi);
      int last = block_n_chi[block_type_index] + 1;
      Vec<Int, 4> tor0_ats = dihedral_atom_inds[pose_index][block_index][0];
      Vec<Int, 4> tor1_ats = dihedral_atom_inds[pose_index][block_index][1];
      Vec<Int, 4> tor2_ats = dihedral_atom_inds[pose_index][block_index][last];
      for (int j = 0; j < 4; ++j) {
        accumulate<D, Vec<Real, 3>>::add(dV_dx[2][pose_index][tor0_ats[j]],
      dneglnprob_nonrot_dtor_xyz[pose_index][block_index][0].row(j));
        accumulate<D, Vec<Real, 3>>::add(dV_dx[2][pose_index][tor1_ats[j]],
      dneglnprob_nonrot_dtor_xyz[pose_index][block_index][1].row(j));
        accumulate<D, Vec<Real, 3>>::add(dV_dx[2][pose_index][tor2_ats[j]],
      dneglnprob_nonrot_dtor_xyz[pose_index][block_index][2].row(j));
      }*/
    }
  });

  DeviceDispatch<D>::forall_stacks(n_poses, max_n_blocks, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
