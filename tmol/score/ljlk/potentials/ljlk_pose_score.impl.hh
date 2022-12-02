#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <int TILE, template <typename> typename InterEnergyData, typename Real>
EIGEN_DEVICE_FUNC int interres_count_pair_separation(
    InterEnergyData<Real> const &inter_dat,
    int atom_tile_ind1,
    int atom_tile_ind2) {
  int separation = inter_dat.min_separation;
  if (separation <= inter_dat.max_important_bond_separation) {
    separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
        inter_dat.max_important_bond_separation,
        atom_tile_ind1,
        atom_tile_ind2,
        inter_dat.r1.n_conn,
        inter_dat.r2.n_conn,
        inter_dat.r1.path_dist,
        inter_dat.r2.path_dist,
        inter_dat.conn_seps);
  }
  return separation;
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,

    // dims: n-poses x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-poses x max-n-blocks x max-n-blocks x
    // max-n-interblock-connections x max-n-interblock-connections
    TView<Int, 5, D> pose_stack_inter_block_bondsep,

    //////////////////////
    // Chemical properties
    // how many atoms for a given block
    // Dimsize n_block_types
    TView<Int, 1, D> block_type_n_atoms,

    TView<Int, 2, D> block_type_n_heavy_atoms_in_tile,
    TView<Int, 2, D> block_type_heavy_atoms_in_tile,

    // what are the atom types for these atoms
    // Dimsize: n_block_types x max_n_atoms
    TView<Int, 2, D> block_type_atom_types,

    // how many inter-block chemical bonds are there
    // Dimsize: n_block_types
    TView<Int, 1, D> block_type_n_interblock_bonds,

    // what atoms form the inter-block chemical bonds
    // Dimsize: n_block_types x max_n_interblock_bonds
    TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,

    // what is the path distance between pairs of atoms in the block
    // Dimsize: n_block_types x max_n_atoms x max_n_atoms
    TView<Int, 3, D> block_type_path_distance,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,
    bool compute_derivs

    ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_poses = coords.size(0);
  int const max_n_pose_atoms = coords.size(1);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);
  int64_t const n_atom_types = type_params.size(0);

  assert(max_n_interblock_bonds <= MAX_N_CONN);

  assert(pose_stack_block_type.size(0) == n_poses);

  assert(pose_stack_min_bond_separation.size(0) == n_poses);
  assert(pose_stack_min_bond_separation.size(1) == max_n_blocks);
  assert(pose_stack_min_bond_separation.size(2) == max_n_blocks);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(2) == max_n_blocks);
  assert(pose_stack_inter_block_bondsep.size(3) == max_n_interblock_bonds);
  assert(pose_stack_inter_block_bondsep.size(4) == max_n_interblock_bonds);

  assert(block_type_n_heavy_atoms_in_tile.size(0) == n_block_types);

  assert(block_type_heavy_atoms_in_tile.size(0) == n_block_types);
  assert(block_type_heavy_atoms_in_tile.size(1) == TILE_SIZE * max_n_tiles);

  assert(block_type_atom_types.size(0) == n_block_types);
  assert(block_type_atom_types.size(1) == max_n_block_atoms);

  assert(block_type_n_interblock_bonds.size(0) == n_block_types);

  assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);

  assert(block_type_path_distance.size(0) == n_block_types);
  assert(block_type_path_distance.size(1) == max_n_block_atoms);
  assert(block_type_path_distance.size(2) == max_n_block_atoms);

  auto output_t = TPack<Real, 2, D>::zeros({2, n_poses});
  auto output = output_t.view;

  auto dV_dcoords_t =
      TPack<Vec<Real, 3>, 3, D>::zeros({2, n_poses, max_n_pose_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_block_spheres_t =
      TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4});
  auto scratch_block_spheres = scratch_block_spheres_t.view;

  auto scratch_block_neighbors_t =
      TPack<Int, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto scratch_block_neighbors = scratch_block_neighbors_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_energies_and_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto lj_atom_energy_and_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const &score_dat,
             int cp_separation) {
          return lj_atom_energy_and_derivs_full(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dV_dcoords  // pass in lambda-captured tensor
          );
        });

    auto lk_atom_energy_and_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const &score_dat,
             int cp_separation) {
          return lk_atom_energy_and_derivs_full(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dV_dcoords  // pass in lambda-captured tensor
          );
        });

    auto score_inter_lj_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int start_atom1,
             int start_atom2,
             int atom_tile_ind1,
             int atom_tile_ind2,
             LJLKScoringData<Real> const &inter_dat) {
          int separation = interres_count_pair_separation<TILE_SIZE>(
              inter_dat, atom_tile_ind1, atom_tile_ind2);
          return lj_atom_energy_and_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              inter_dat,
              separation);
        });

    auto score_intra_lj_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int start_atom1,
             int start_atom2,
             int atom_tile_ind1,
             int atom_tile_ind2,
             LJLKScoringData<Real> const &intra_dat) {
          int const atom_ind1 = start_atom1 + atom_tile_ind1;
          int const atom_ind2 = start_atom2 + atom_tile_ind2;

          int const separation =

              block_type_path_distance[intra_dat.r1.block_type][atom_ind1]
                                      [atom_ind2];
          return lj_atom_energy_and_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              intra_dat,
              separation);
        });

    auto score_inter_lk_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int start_atom1,
             int start_atom2,
             int atom_heavy_tile_ind1,
             int atom_heavy_tile_ind2,
             LJLKScoringData<Real> const &inter_dat) {
          int const atom_tile_ind1 =
              inter_dat.r1.heavy_inds[atom_heavy_tile_ind1];
          int const atom_tile_ind2 =
              inter_dat.r2.heavy_inds[atom_heavy_tile_ind2];

          int separation = interres_count_pair_separation<TILE_SIZE>(
              inter_dat, atom_tile_ind1, atom_tile_ind2);

          return lk_atom_energy_and_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              inter_dat,
              separation);
        });

    auto score_intra_lk_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int start_atom1,
             int start_atom2,
             int atom_heavy_tile_ind1,
             int atom_heavy_tile_ind2,
             LJLKScoringData<Real> const &intra_dat) {
          int const atom_tile_ind1 =
              intra_dat.r1.heavy_inds[atom_heavy_tile_ind1];
          int const atom_tile_ind2 =
              intra_dat.r2.heavy_inds[atom_heavy_tile_ind2];
          int const atom_ind1 = start_atom1 + atom_tile_ind1;
          int const atom_ind2 = start_atom2 + atom_tile_ind2;

          int const separation =
              block_type_path_distance[intra_dat.r1.block_type][atom_ind1]
                                      [atom_ind2];
          return lk_atom_energy_and_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              intra_dat,
              separation);
        });

    auto load_block_coords_and_params_into_shared =
        ([=] TMOL_DEVICE_FUNC(
             int pose_ind,
             LJLKSingleResData<Real> &r_dat,
             int n_atoms_to_load,
             int start_atom) {
          ljlk_load_block_coords_and_params_into_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_heavy_atoms_in_tile,
              pose_ind,
              r_dat,
              n_atoms_to_load,
              start_atom);
        });

    auto load_block_into_shared = ([=] TMOL_DEVICE_FUNC(
                                       int pose_ind,
                                       LJLKSingleResData<Real> &r_dat,
                                       int n_atoms_to_load,
                                       int start_atom,
                                       bool count_pair_striking_dist,
                                       unsigned char *__restrict__ conn_ats) {
      ljlk_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE>(
          coords,
          block_type_atom_types,
          type_params,
          block_type_heavy_atoms_in_tile,
          block_type_path_distance,
          pose_ind,
          r_dat,
          n_atoms_to_load,
          start_atom,
          count_pair_striking_dist,
          conn_ats);
    });

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    Real total_lj = 0;
    Real total_lk = 0;

    int const pose_ind = cta / (max_n_blocks * max_n_blocks);
    int const block_ind_pair = cta % (max_n_blocks * max_n_blocks);
    int const block_ind1 = block_ind_pair / max_n_blocks;
    int const block_ind2 = block_ind_pair % max_n_blocks;
    if (block_ind1 > block_ind2) {
      return;
    }

    if (scratch_block_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }

    int const max_important_bond_separation = 4;

    int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
    int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=](int pose_ind,
             int block_ind1,
             int block_ind2,
             int block_type1,
             int block_type2,
             int n_atoms1,
             int n_atoms2,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          ljlk_load_tile_invariant_interres_data<DeviceDispatch, D, nt>(
              pose_stack_block_coord_offset,
              pose_stack_min_bond_separation,
              block_type_n_interblock_bonds,
              block_type_atoms_forming_chemical_bonds,
              pose_stack_inter_block_bondsep,
              global_params,
              max_important_bond_separation,
              pose_ind,
              block_ind1,
              block_ind2,
              block_type1,
              block_type2,
              n_atoms1,
              n_atoms2,
              inter_dat,
              shared.m);
        });

    auto load_interres1_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom1,
             int n_atoms_to_load1,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          ljlk_load_interres1_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_heavy_atoms_in_tile,
              block_type_path_distance,
              block_type_n_heavy_atoms_in_tile,
              tile_ind,
              start_atom1,
              n_atoms_to_load1,
              inter_dat,
              shared.m);
        });

    auto load_interres2_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom2,
             int n_atoms_to_load2,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          ljlk_load_interres2_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_heavy_atoms_in_tile,
              block_type_path_distance,
              block_type_n_heavy_atoms_in_tile,
              tile_ind,
              start_atom2,
              n_atoms_to_load2,
              inter_dat,
              shared.m);
        });

    auto load_interres_data_from_shared =
        ([=](int,
             int,
             shared_mem_union &shared,
             LJLKScoringData<Real> &inter_dat) {
          ljlk_load_interres_data_from_shared(shared.m, inter_dat);
        });

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_interres_atom_pair_scores = ([=](LJLKScoringData<Real> &inter_dat,
                                               int start_atom1,
                                               int start_atom2) {
      auto eval_scores_for_atom_pairs = ([&](int tid) {
        inter_dat.total_lj += tmol::score::common::InterResBlockEvaluation<
            LJLKScoringData,
            AllAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_interres_atom_pair(
                tid,
                start_atom1,
                start_atom2,
                score_inter_lj_atom_pair,
                inter_dat);

        inter_dat.total_lk += tmol::score::common::InterResBlockEvaluation<
            LJLKScoringData,
            HeavyAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_interres_atom_pair(
                tid,
                start_atom1,
                start_atom2,
                score_inter_lk_atom_pair,
                inter_dat);
      });

      // The work: On GPU threads work independently, on CPU, this will be a
      // for loop
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_scores_for_atom_pairs);
    });

    auto store_calculated_energies = ([=](LJLKScoringData<Real> &score_dat,
                                          shared_mem_union &shared) {
      auto reduce_energies = ([&](int tid) {
        Real const cta_total_lj =
            DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
                score_dat.total_lj, shared, mgpu::plus_t<Real>());
        Real const cta_total_lk =
            DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
                score_dat.total_lk, shared, mgpu::plus_t<Real>());

        if (tid == 0) {
          // printf("Storing energy %d %d %f %f\n", score_dat.block_ind1,
          // score_dat.block_ind2, cta_total_lj, cta_total_lk);
          accumulate<D, Real>::add(output[0][score_dat.pose_ind], cta_total_lj);
          accumulate<D, Real>::add(output[1][score_dat.pose_ind], cta_total_lk);
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
    });

    auto load_tile_invariant_intrares_data =
        ([=](int pose_ind,
             int block_ind1,
             int block_type1,
             int n_atoms1,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          ljlk_load_tile_invariant_intrares_data<DeviceDispatch, D, nt>(
              pose_stack_block_coord_offset,
              global_params,
              max_important_bond_separation,
              pose_ind,
              block_ind1,
              block_type1,
              n_atoms1,
              intra_dat,
              shared.m);
        });

    auto load_intrares1_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom1,
             int n_atoms_to_load1,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          ljlk_load_intrares1_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_n_heavy_atoms_in_tile,
              block_type_heavy_atoms_in_tile,
              tile_ind,
              start_atom1,
              n_atoms_to_load1,
              intra_dat,
              shared.m);
        });

    auto load_intrares2_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom2,
             int n_atoms_to_load2,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          ljlk_load_intrares2_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_n_heavy_atoms_in_tile,
              block_type_heavy_atoms_in_tile,
              tile_ind,
              start_atom2,
              n_atoms_to_load2,
              intra_dat,
              shared.m);
        });

    auto load_intrares_data_from_shared =
        ([=](int tile_ind1,
             int tile_ind2,
             shared_mem_union &shared,
             LJLKScoringData<Real> &intra_dat) {
          ljlk_load_intrares_data_from_shared(
              tile_ind1, tile_ind2, shared.m, intra_dat);
        });

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_intrares_atom_pair_scores = ([=](LJLKScoringData<Real> &intra_dat,
                                               int start_atom1,
                                               int start_atom2) {
      auto eval_scores_for_atom_pairs = ([&](int tid) {
        intra_dat.total_lj += tmol::score::common::IntraResBlockEvaluation<
            LJLKScoringData,
            AllAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_intrares_atom_pairs(
                tid,
                start_atom1,
                start_atom2,
                score_intra_lj_atom_pair,
                intra_dat);
        intra_dat.total_lk += tmol::score::common::IntraResBlockEvaluation<
            LJLKScoringData,
            HeavyAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_intrares_atom_pairs(
                tid,
                start_atom1,
                start_atom2,
                score_intra_lk_atom_pair,
                intra_dat);
      });

      // The work: On GPU threads work independently, on CPU, this will be a
      // for loop
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_scores_for_atom_pairs);
    });

    tmol::score::common::tile_evaluate_block_pair<
        DeviceDispatch,
        D,
        LJLKScoringData,
        LJLKScoringData,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  auto eval_energies_only = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto score_inter_lj_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int,
             int,
             int atom_tile_ind1,
             int atom_tile_ind2,
             LJLKScoringData<Real> const &inter_dat) {
          int separation = interres_count_pair_separation<TILE_SIZE>(
              inter_dat, atom_tile_ind1, atom_tile_ind2);
          return lj_atom_energy(
              atom_tile_ind1, atom_tile_ind2, inter_dat, separation);
        });

    auto score_intra_lj_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int start_atom1,
             int start_atom2,
             int atom_tile_ind1,
             int atom_tile_ind2,
             LJLKScoringData<Real> const &intra_dat) {
          int const atom_ind1 = start_atom1 + atom_tile_ind1;
          int const atom_ind2 = start_atom2 + atom_tile_ind2;

          int const separation =

              block_type_path_distance[intra_dat.r1.block_type][atom_ind1]
                                      [atom_ind2];
          return lj_atom_energy(
              atom_tile_ind1, atom_tile_ind2, intra_dat, separation);
        });

    auto score_inter_lk_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int,
             int,
             int atom_heavy_tile_ind1,
             int atom_heavy_tile_ind2,
             LJLKScoringData<Real> const &inter_dat) {
          int const atom_tile_ind1 =
              inter_dat.r1.heavy_inds[atom_heavy_tile_ind1];
          int const atom_tile_ind2 =
              inter_dat.r2.heavy_inds[atom_heavy_tile_ind2];

          int separation = interres_count_pair_separation<TILE_SIZE>(
              inter_dat, atom_tile_ind1, atom_tile_ind2);

          return lk_atom_energy(
              atom_tile_ind1, atom_tile_ind2, inter_dat, separation);
        });

    auto score_intra_lk_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int start_atom1,
             int start_atom2,
             int atom_heavy_tile_ind1,
             int atom_heavy_tile_ind2,
             LJLKScoringData<Real> const &intra_dat) {
          int const atom_tile_ind1 =
              intra_dat.r1.heavy_inds[atom_heavy_tile_ind1];
          int const atom_tile_ind2 =
              intra_dat.r2.heavy_inds[atom_heavy_tile_ind2];
          int const atom_ind1 = start_atom1 + atom_tile_ind1;
          int const atom_ind2 = start_atom2 + atom_tile_ind2;

          int const separation =
              block_type_path_distance[intra_dat.r1.block_type][atom_ind1]
                                      [atom_ind2];
          return lk_atom_energy(
              atom_tile_ind1, atom_tile_ind2, intra_dat, separation);
        });

    auto load_block_coords_and_params_into_shared =
        ([=] TMOL_DEVICE_FUNC(
             int pose_ind,
             LJLKSingleResData<Real> &r_dat,
             int n_atoms_to_load,
             int start_atom) {
          ljlk_load_block_coords_and_params_into_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_heavy_atoms_in_tile,
              pose_ind,
              r_dat,
              n_atoms_to_load,
              start_atom);
        });

    auto load_block_into_shared = ([=] TMOL_DEVICE_FUNC(
                                       int pose_ind,
                                       LJLKSingleResData<Real> &r_dat,
                                       int n_atoms_to_load,
                                       int start_atom,
                                       bool count_pair_striking_dist,
                                       unsigned char *__restrict__ conn_ats) {
      ljlk_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE>(
          coords,
          block_type_atom_types,
          type_params,
          block_type_heavy_atoms_in_tile,
          block_type_path_distance,
          pose_ind,
          r_dat,
          n_atoms_to_load,
          start_atom,
          count_pair_striking_dist,
          conn_ats);
    });

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    Real total_lj = 0;
    Real total_lk = 0;

    int const pose_ind = cta / (max_n_blocks * max_n_blocks);
    int const block_ind_pair = cta % (max_n_blocks * max_n_blocks);
    int const block_ind1 = block_ind_pair / max_n_blocks;
    int const block_ind2 = block_ind_pair % max_n_blocks;
    if (block_ind1 > block_ind2) {
      return;
    }

    if (scratch_block_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }

    int const max_important_bond_separation = 4;

    int const block_type1 = pose_stack_block_type[pose_ind][block_ind1];
    int const block_type2 = pose_stack_block_type[pose_ind][block_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=](int pose_ind,
             int block_ind1,
             int block_ind2,
             int block_type1,
             int block_type2,
             int n_atoms1,
             int n_atoms2,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          ljlk_load_tile_invariant_interres_data<DeviceDispatch, D, nt>(
              pose_stack_block_coord_offset,
              pose_stack_min_bond_separation,
              block_type_n_interblock_bonds,
              block_type_atoms_forming_chemical_bonds,
              pose_stack_inter_block_bondsep,
              global_params,
              max_important_bond_separation,
              pose_ind,
              block_ind1,
              block_ind2,
              block_type1,
              block_type2,
              n_atoms1,
              n_atoms2,
              inter_dat,
              shared.m);
        });

    auto load_interres1_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom1,
             int n_atoms_to_load1,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          ljlk_load_interres1_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_heavy_atoms_in_tile,
              block_type_path_distance,
              block_type_n_heavy_atoms_in_tile,
              tile_ind,
              start_atom1,
              n_atoms_to_load1,
              inter_dat,
              shared.m);
        });

    auto load_interres2_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom2,
             int n_atoms_to_load2,
             LJLKScoringData<Real> &inter_dat,
             shared_mem_union &shared) {
          ljlk_load_interres2_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_heavy_atoms_in_tile,
              block_type_path_distance,
              block_type_n_heavy_atoms_in_tile,
              tile_ind,
              start_atom2,
              n_atoms_to_load2,
              inter_dat,
              shared.m);
        });

    auto load_interres_data_from_shared =
        ([=](int,
             int,
             shared_mem_union &shared,
             LJLKScoringData<Real> &inter_dat) {
          ljlk_load_interres_data_from_shared(shared.m, inter_dat);
        });

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_interres_atom_pair_scores = ([=](LJLKScoringData<Real> &inter_dat,
                                               int start_atom1,
                                               int start_atom2) {
      auto eval_scores_for_atom_pairs = ([&](int tid) {
        inter_dat.total_lj += tmol::score::common::InterResBlockEvaluation<
            LJLKScoringData,
            AllAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_interres_atom_pair(
                tid,
                start_atom1,
                start_atom2,
                score_inter_lj_atom_pair,
                inter_dat);

        inter_dat.total_lk += tmol::score::common::InterResBlockEvaluation<
            LJLKScoringData,
            HeavyAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_interres_atom_pair(
                tid,
                start_atom1,
                start_atom2,
                score_inter_lk_atom_pair,
                inter_dat);
      });

      // The work: On GPU threads work independently, on CPU, this will be a
      // for loop
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_scores_for_atom_pairs);
    });

    auto store_calculated_energies = ([=](LJLKScoringData<Real> &score_dat,
                                          shared_mem_union &shared) {
      auto reduce_energies = ([&](int tid) {
        Real const cta_total_lj =
            DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
                score_dat.total_lj, shared, mgpu::plus_t<Real>());
        Real const cta_total_lk =
            DeviceDispatch<D>::template reduce_in_workgroup<TILE_SIZE>(
                score_dat.total_lk, shared, mgpu::plus_t<Real>());

        if (tid == 0) {
          // printf("Storing energy %d %d %f %f\n", score_dat.block_ind1,
          // score_dat.block_ind2, cta_total_lj, cta_total_lk);
          accumulate<D, Real>::add(output[0][score_dat.pose_ind], cta_total_lj);
          accumulate<D, Real>::add(output[1][score_dat.pose_ind], cta_total_lk);
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
    });

    auto load_tile_invariant_intrares_data =
        ([=](int pose_ind,
             int block_ind1,
             int block_type1,
             int n_atoms1,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          ljlk_load_tile_invariant_intrares_data<DeviceDispatch, D, nt>(
              pose_stack_block_coord_offset,
              global_params,
              max_important_bond_separation,
              pose_ind,
              block_ind1,
              block_type1,
              n_atoms1,
              intra_dat,
              shared.m);
        });

    auto load_intrares1_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom1,
             int n_atoms_to_load1,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          ljlk_load_intrares1_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_n_heavy_atoms_in_tile,
              block_type_heavy_atoms_in_tile,
              tile_ind,
              start_atom1,
              n_atoms_to_load1,
              intra_dat,
              shared.m);
        });

    auto load_intrares2_tile_data_to_shared =
        ([=](int tile_ind,
             int start_atom2,
             int n_atoms_to_load2,
             LJLKScoringData<Real> &intra_dat,
             shared_mem_union &shared) {
          ljlk_load_intrares2_tile_data_to_shared<DeviceDispatch, D, nt>(
              coords,
              block_type_atom_types,
              type_params,
              block_type_n_heavy_atoms_in_tile,
              block_type_heavy_atoms_in_tile,
              tile_ind,
              start_atom2,
              n_atoms_to_load2,
              intra_dat,
              shared.m);
        });

    auto load_intrares_data_from_shared =
        ([=](int tile_ind1,
             int tile_ind2,
             shared_mem_union &shared,
             LJLKScoringData<Real> &intra_dat) {
          ljlk_load_intrares_data_from_shared(
              tile_ind1, tile_ind2, shared.m, intra_dat);
        });

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_intrares_atom_pair_scores = ([=](LJLKScoringData<Real> &intra_dat,
                                               int start_atom1,
                                               int start_atom2) {
      auto eval_scores_for_atom_pairs = ([&](int tid) {
        intra_dat.total_lj += tmol::score::common::IntraResBlockEvaluation<
            LJLKScoringData,
            AllAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_intrares_atom_pairs(
                tid,
                start_atom1,
                start_atom2,
                score_intra_lj_atom_pair,
                intra_dat);
        intra_dat.total_lk += tmol::score::common::IntraResBlockEvaluation<
            LJLKScoringData,
            HeavyAtomPairSelector,
            D,
            TILE_SIZE,
            nt,
            Real,
            Int>::
            eval_intrares_atom_pairs(
                tid,
                start_atom1,
                start_atom2,
                score_intra_lk_atom_pair,
                intra_dat);
      });

      // The work: On GPU threads work independently, on CPU, this will be a
      // for loop
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_scores_for_atom_pairs);
    });

    tmol::score::common::tile_evaluate_block_pair<
        DeviceDispatch,
        D,
        LJLKScoringData,
        LJLKScoringData,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  ///////////////////////////////////////////////////////////////////////

  // Three steps
  // 0: setup
  // 1: launch a kernel to find a small bounding sphere surrounding the blocks
  // 2: launch a kernel to look for spheres that are within striking distance of
  // each other
  // 3: launch a kernel to evaluate lj/lk between pairs of blocks
  // within striking distance

  // 0
  // TO DO: let DeviceDispatch hold a cuda stream (??)
  // at::cuda::CUDAStream wrapped_stream = at::cuda::getDefaultCUDAStream();
  // mgpu::standard_context_t context(wrapped_stream.stream());
  int const n_block_pairs = n_poses * max_n_blocks * max_n_blocks;

  score::common::sphere_overlap::
      compute_block_spheres<DeviceDispatch, D, Real, Int>::f(
          coords,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          block_type_n_atoms,
          scratch_block_spheres);

  score::common::sphere_overlap::
      detect_block_neighbors<DeviceDispatch, D, Real, Int>::f(
          coords,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          block_type_n_atoms,
          scratch_block_spheres,
          scratch_block_neighbors,
          Real(6.0));  // 6A hard coded here. Please fix! TEMP!

  // 3
  if (compute_derivs) {
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        n_block_pairs, eval_energies_and_derivs);
  } else {
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        n_block_pairs, eval_energies_only);
  }

  return {output_t, dV_dcoords_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
