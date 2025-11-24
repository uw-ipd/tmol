#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/ljlk_scoring_macros.hh>
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

// template <int TILE, template <typename> typename InterEnergyData, typename
// Real> EIGEN_DEVICE_FUNC int interres_count_pair_separation(
//     InterEnergyData<Real> const& inter_dat,
//     int atom_tile_ind1,
//     int atom_tile_ind2) {
//   int separation = inter_dat.min_separation;
//   if (separation <= inter_dat.max_important_bond_separation) {
//     separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
//         inter_dat.max_important_bond_separation,
//         atom_tile_ind1,
//         atom_tile_ind2,
//         inter_dat.r1.n_conn,
//         inter_dat.r2.n_conn,
//         inter_dat.r1.path_dist,
//         inter_dat.r2.path_dist,
//         inter_dat.conn_seps);
//   }
//   return separation;
// }

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::forward(
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

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
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

    // should the output be per-pose (npose x nterms x 1 x 1)
    //   or per block-pair (npose x nterms x len x len)
    bool output_block_pair_energies,
    bool require_gradient) -> std::
    tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 3, D> > {
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= MAX_N_CONN);

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

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_rot_spheres_t =
      TPack<Real, 3, D>::zeros({n_poses, max_n_rots_per_pose, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_rot_neighbors_t = TPack<Int, 3, D>::zeros(
      {n_poses, max_n_rots_per_pose, max_n_rots_per_pose});
  auto scratch_rot_neighbors = scratch_rot_neighbors_t.view;

  // TPack<Int, 2, Dev> dispatch_indices_t;

  // score::common::sphere_overlap::
  //     compute_rot_spheres<DeviceOperations, D, Real, Int>::f(
  //         rot_coords,
  //         rot_coord_offset,
  //         block_type_ind_for_rot,
  //         block_type_n_atoms,
  //         scratch_rot_spheres);

  // score::common::sphere_overlap::
  //     detect_rot_neighbors<DeviceOperations, D, Real, Int>::f(
  //         max_n_rots_per_pose,
  //         block_ind_for_rot,
  //         block_type_ind_for_rot,
  //         block_type_n_atoms,
  //         n_rots_for_pose,
  //         rot_offset_for_pose,
  //         n_rots_for_block,
  //         scratch_rot_spheres,
  //         scratch_rot_neighbors,
  //         Real(5.5));  // 5.5A hard coded here. Please fix! TEMP!

  // auto dispatch_indices_t = score::common::sphere_overlap::
  //     rot_neighbor_indices<DeviceOperations, D, Int>::f(
  //         scratch_rot_neighbors, rot_offset_for_pose);
  // auto rni_result = score::common::sphere_overlap::
  //     rot_neighbor_indices<DeviceOperations, D, Int>::f(
  //         scratch_rot_neighbors, rot_offset_for_pose);
  // auto dispatch_indices_t = std::get<0>(rni_result);
  // auto offset_for_cell_t = std::get<1>(rni_result);

  // auto dispatch_indices = dispatch_indices_t.view;

  // TPack<Real, 2, D> output_t;
  // if (output_block_pair_energies) {
  //   output_t = TPack<Real, 2, D>::zeros({3, dispatch_indices.size(1)});
  // } else {
  //   // printf("n poses for whole-pose scoring? %d\n", n_poses);
  //   output_t = TPack<Real, 2, D>::zeros({3, n_poses});
  // }
  // auto output = output_t.view;
  TPack<Real, 4, D> output_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, D>::zeros({3, n_poses, 1, 1});
  }
  auto output = output_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;
  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = (max_n_blocks * (max_n_blocks + 1)) / 2;

  // There are two versions of scoring:
  // Block-pair scoring, where the output is written to an n-pose x n-blocks x
  // n-blocks tensor, and whole-pose scoring, where the output is
  // atomic-incremented into an n-pose x 1 x 1 tensor. In the whole-pose scoring
  // case, we compute the atomic derivatives in the forward pass and then weight
  // them by whatever weights are applied to the terms; in a very mininmal
  // backward pass (see compiled_ops.cpp). Thread 0 in each CTA atomic- adds the
  // block-pair energy for the block pair it is assigned to. In the block-pair
  // scoring case, we are unable to compute the atomic derivatives in the
  // forward pass, because how the block-pair outputs will be weighted is not
  // known until the backward pass and the memory to store atom-pair derivatives
  // for each interacting block pair would be prohibitive. So we do not compute
  // derivatives at all in the forward pass. It is slightly more efficient to
  // have a separate kernel that bypasses the derivative calculations but the
  // actual performance difference for having a second kernel has not been
  // measured. (TO DO, go ahead and measure it!)
  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int,
                                int,
                                LJLKScoringData<Real> const& score_dat,
                                int cp_separation) {
      return lj_atom_energy(
          atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
    });

    auto atom_pair_lk_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int,
                                int,
                                LJLKScoringData<Real> const& score_dat,
                                int cp_separation) {
      return lk_atom_energy(
          atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
    });

    auto score_inter_lj_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_intra_lj_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_inter_lk_atom_pair =
        ([=] SCORE_INTER_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto score_intra_lk_atom_pair =
        ([=] SCORE_INTRA_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    Real total_ljatr = 0;
    Real total_ljrep = 0;
    Real total_lk = 0;

    int const max_important_bond_separation = 4;

    int const pose_ind = cta / (max_n_upper_triangle_inds);
    int const block_ind_pair = cta % (max_n_upper_triangle_inds);

    // We do not have to kill half of our thread blocks simply because they
    // represent the lower triangle now that we're using upper-triangle indices
    auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
        block_ind_pair, max_n_blocks + 1);

    int const block_ind1 = common::get<0>(upper_triangle_ind);
    int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

    // We still kill CTAs targetting non-neighboring block pairs, though,
    // and that can be a lot
    if (scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }
    int const rot_ind1 = rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = rot_offset_for_block[pose_ind][block_ind2];

    if (rot_ind1 < 0 || rot_ind2 < 0) {
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_POSE_CALCULATED_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceOperations,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
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

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int start_atom1,
                                int start_atom2,
                                LJLKScoringData<Real> const& score_dat,
                                int cp_separation) {
      if (require_gradient) {  // captured
        return lj_atom_energy_and_derivs_full(
            atom_tile_ind1,
            atom_tile_ind2,
            start_atom1,
            start_atom2,
            score_dat,
            cp_separation,
            dV_dcoords  // captured
        );
      } else {
        return lj_atom_energy(
            atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
      }
    });

    auto atom_pair_lk_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int start_atom1,
                                int start_atom2,
                                LJLKScoringData<Real> const& score_dat,
                                int cp_separation) {
      if (require_gradient) {  // captured
        return lk_atom_energy_and_derivs_full(
            atom_tile_ind1,
            atom_tile_ind2,
            start_atom1,
            start_atom2,
            score_dat,
            cp_separation,
            dV_dcoords  // captured
        );
      } else {
        return lk_atom_energy(
            atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
      }
    });

    auto score_inter_lj_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_intra_lj_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_inter_lk_atom_pair =
        ([=] SCORE_INTER_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto score_intra_lk_atom_pair =
        ([=] SCORE_INTRA_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    Real total_lj = 0;
    Real total_lk = 0;

    int const max_important_bond_separation = 4;

    int const pose_ind = cta / (max_n_upper_triangle_inds);
    int const block_ind_pair = cta % (max_n_upper_triangle_inds);

    // We do not have to kill half of our thread blocks simply because they
    // represent the lower triangle now that we're using upper-triangle indices
    auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
        block_ind_pair, max_n_blocks + 1);

    int const block_ind1 = common::get<0>(upper_triangle_ind);
    int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

    // We still kill CTAs targetting non-neighboring block pairs, though,
    // and that can be a lot
    if (scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }
    int const rot_ind1 = rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = rot_offset_for_block[pose_ind][block_ind2];

    if (rot_ind1 < 0 || rot_ind2 < 0) {
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }
    // printf("scoring pose %d, block pair (%d, %d) rotamers (%d, %d) types (%d,
    // %d)\n",
    //        pose_ind,
    //        block_ind1,
    //        block_ind2,
    //        rot_ind1,
    //        rot_ind2,
    //        block_type1,
    //        block_type2);

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_POSE_CALCULATED_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceOperations,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
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
  score::common::sphere_overlap::
      compute_block_spheres<DeviceOperations, D, Real, Int>::f(
          rot_coords,
          rot_coord_offset,
          block_ind_for_rot,
          pose_ind_for_rot,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres);
  // printf("computed block spheres\n");

  score::common::sphere_overlap::
      detect_block_neighbors<DeviceOperations, D, Real, Int>::f(
          first_rot_block_type,
          scratch_rot_spheres,
          scratch_rot_neighbors,
          Real(5.5));
  // printf("detected block neighbors\n");

  if (output_block_pair_energies) {
    DeviceOperations<D>::template foreach_workgroup<launch_t>(
        n_poses * max_n_upper_triangle_inds, eval_energies_by_block);
  } else {
    DeviceOperations<D>::template foreach_workgroup<launch_t>(
        n_poses * max_n_upper_triangle_inds, eval_energies);
  }
  // printf("evaluated energies\n");
  // DeviceOperations<D>::synchronize_device();

  return {output_t, dV_dcoords_t, scratch_rot_neighbors_t};
}  // LJLKPoseScoreDispatch::forward

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::backward(
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

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
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

    TView<Int, 3, D> scratch_rot_neighbors,  // from forward pass

    TView<Real, 4, D> dTdV  // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 1, D> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= MAX_N_CONN);

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

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 1, D>::zeros({n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;
  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = (max_n_blocks * (max_n_blocks + 1)) / 2;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const& score_dat,
             int cp_separation) -> std::array<Real, 2> {
          lj_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV[0][score_dat.pose_ind][score_dat.block_ind1]
                  [score_dat.block_ind2],
              dTdV[1][score_dat.pose_ind][score_dat.block_ind1]
                  [score_dat.block_ind2],
              dV_dcoords  // captured
          );
          return {0.0, 0.0};
        });

    auto atom_pair_lk_fn =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const& score_dat,
             int cp_separation) -> Real {
          lk_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV[2][score_dat.pose_ind][score_dat.block_ind1]
                  [score_dat.block_ind2],
              dV_dcoords  // captured
          );
          return 0.0;
        });

    auto score_inter_lj_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_intra_lj_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_inter_lk_atom_pair =
        ([=] SCORE_INTER_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto score_intra_lk_atom_pair =
        ([=] SCORE_INTRA_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    Real total_lj = 0;
    Real total_lk = 0;

    int const max_important_bond_separation = 4;

    int const pose_ind = cta / (max_n_upper_triangle_inds);
    int const block_ind_pair = cta % (max_n_upper_triangle_inds);

    auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
        block_ind_pair, max_n_blocks + 1);
    int const block_ind1 = common::get<0>(upper_triangle_ind);
    int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

    int const rot_ind1 = rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = rot_offset_for_block[pose_ind][block_ind2];

    if (scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies =
        ([=](LJLKScoringData<Real>& score_dat, shared_mem_union& shared) {
          ;  // no op for gradients ()
        });

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceOperations,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
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

  DeviceOperations<D>::template foreach_workgroup<launch_t>(
      n_poses * max_n_upper_triangle_inds, eval_derivs);

  return dV_dcoords_t;
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRotamerScoreDispatch<DeviceOperations, D, Real, Int>::forward(
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

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
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

    // should the output be per-pose (npose x nterms x 1 x 1)
    //   or per block-pair (npose x nterms x len x len)
    bool output_block_pair_energies,
    bool require_gradient) -> std::
    tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D> > {
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= MAX_N_CONN);

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

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  auto scratch_rot_spheres_t = TPack<Real, 2, D>::zeros({n_rots, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_rot_neighbors_t = TPack<Int, 3, D>::zeros(
      {n_poses, max_n_rots_per_pose, max_n_rots_per_pose});
  auto scratch_rot_neighbors = scratch_rot_neighbors_t.view;

  // TPack<Int, 2, Dev> dispatch_indices_t;

  score::common::sphere_overlap::
      compute_rot_spheres<DeviceOperations, D, Real, Int>::f(
          rot_coords,
          rot_coord_offset,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres);

  score::common::sphere_overlap::
      detect_rot_neighbors<DeviceOperations, D, Real, Int>::f(
          max_n_rots_per_pose,
          block_ind_for_rot,
          block_type_ind_for_rot,
          block_type_n_atoms,
          n_rots_for_pose,
          rot_offset_for_pose,
          n_rots_for_block,
          scratch_rot_spheres,
          scratch_rot_neighbors,
          Real(5.5));  // 5.5A hard coded here. Please fix! TEMP!

  auto dispatch_indices_t = score::common::sphere_overlap::
      rot_neighbor_indices<DeviceOperations, D, Int>::f(
          scratch_rot_neighbors, rot_offset_for_pose);
  // auto rni_result = score::common::sphere_overlap::
  //     rot_neighbor_indices<DeviceOperations, D, Int>::f(
  //         scratch_rot_neighbors, rot_offset_for_pose);
  // auto dispatch_indices_t = std::get<0>(rni_result);
  // auto offset_for_cell_t = std::get<1>(rni_result);

  auto dispatch_indices = dispatch_indices_t.view;

  TPack<Real, 2, D> output_t;
  if (output_block_pair_energies) {
    output_t = TPack<Real, 2, D>::zeros({3, dispatch_indices.size(1)});
  } else {
    // printf("n poses for whole-pose scoring? %d\n", n_poses);
    output_t = TPack<Real, 2, D>::zeros({3, n_poses});
  }
  auto output = output_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int,
                                int,
                                LJLKScoringData<Real> const& score_dat,
                                int cp_separation) {
      return lj_atom_energy(
          atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
    });

    auto atom_pair_lk_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int,
                                int,
                                LJLKScoringData<Real> const& score_dat,
                                int cp_separation) {
      return lk_atom_energy(
          atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
    });

    auto score_inter_lj_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_intra_lj_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_inter_lk_atom_pair =
        ([=] SCORE_INTER_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto score_intra_lk_atom_pair =
        ([=] SCORE_INTRA_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    Real total_ljatr = 0;
    Real total_ljrep = 0;
    Real total_lk = 0;

    int const max_important_bond_separation = 4;

    int const pose_ind = dispatch_indices[0][cta];

    int const rot_ind1 = dispatch_indices[1][cta];
    int const rot_ind2 = dispatch_indices[2][cta];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_ROTAMER_CALCULATED_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceOperations,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
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

  assert(output_block_pair_energies);
  DeviceOperations<D>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_energies_by_block);
  // DeviceOperations<D>::synchronize_device();

  return {output_t, dV_dcoords_t, dispatch_indices_t};
}  // LJLKRotamerScoreDispatch::forward

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRotamerScoreDispatch<DeviceOperations, D, Real, Int>::backward(
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

    // dims: n-systems x max-n-blocks x max-n-blocks
    // Quick lookup: given the inds of two blocks, ask: what is the minimum
    // number of chemical bonds that separate any pair of atoms in those
    // blocks? If this minimum is greater than the crossover, then no further
    // logic for deciding whether two atoms in those blocks should have their
    // interaction energies calculated: all should. intentionally small to
    // (possibly) fit in constant cache
    TView<Int, 3, D> pose_stack_min_bond_separation,

    // dims: n-systems x max-n-blocks x max-n-blocks x
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

    TView<Int, 2, D> dispatch_indices,  // from forward pass

    TView<Real, 2, D> dTdV  // nterms x n-dispatch
    ) -> TPack<Vec<Real, 3>, 1, D> {
  using tmol::score::common::accumulate;
  using Real3 = Vec<Real, 3>;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_block_types = block_type_n_atoms.size(0);
  int const max_n_block_atoms = block_type_atom_types.size(1);
  int const max_n_interblock_bonds =
      block_type_atoms_forming_chemical_bonds.size(1);

  int const max_n_tiles = block_type_n_heavy_atoms_in_tile.size(2);
  int64_t const n_atom_types = type_params.size(0);

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

  assert(max_n_interblock_bonds <= MAX_N_CONN);

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

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 1, D>::zeros({n_atoms});
  auto dV_dcoords = dV_dcoords_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const& score_dat,
             int cp_separation) -> std::array<Real, 2> {
          lj_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV[0][cta],
              dTdV[1][cta],
              dV_dcoords  // captured
          );
          return {0.0, 0.0};
        });

    auto atom_pair_lk_fn =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const& score_dat,
             int cp_separation) -> Real {
          lk_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV[2][cta],
              dV_dcoords  // captured
          );
          return 0.0;
        });

    auto score_inter_lj_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_intra_lj_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_inter_lk_atom_pair =
        ([=] SCORE_INTER_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto score_intra_lk_atom_pair =
        ([=] SCORE_INTRA_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

    Real total_lj = 0;
    Real total_lk = 0;

    int const max_important_bond_separation = 4;

    int const pose_ind = dispatch_indices[0][cta];

    int const rot_ind1 = dispatch_indices[1][cta];
    int const rot_ind2 = dispatch_indices[2][cta];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies =
        ([=](LJLKScoringData<Real>& score_dat, shared_mem_union& shared) {
          ;  // no op for gradients ()
        });

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceOperations,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
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

  DeviceOperations<D>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_derivs);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
