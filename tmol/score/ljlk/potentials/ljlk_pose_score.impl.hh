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

//// kernel macros
//    these define functions that are used in multiple lambda captures
//    variables that are expected to be captured for each macro are specified

// SCORE_INTER_LJ_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->std::array<Real, 2>
#define SCORE_INTER_LJ_ATOM_PAIR(atom_pair_func)                \
  TMOL_DEVICE_FUNC(                                             \
      int start_atom1,                                          \
      int start_atom2,                                          \
      int atom_tile_ind1,                                       \
      int atom_tile_ind2,                                       \
      LJLKScoringData<Real> const &inter_dat) {                 \
    int separation = interres_count_pair_separation<TILE_SIZE>( \
        inter_dat, atom_tile_ind1, atom_tile_ind2);             \
    return atom_pair_func(                                      \
        atom_tile_ind1,                                         \
        atom_tile_ind2,                                         \
        start_atom1,                                            \
        start_atom2,                                            \
        inter_dat,                                              \
        separation);                                            \
  }

// SCORE_INTRA_LJ_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->std::array<Real, 2>
// captures:
//    block_type_path_distance
#define SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_func)                             \
  TMOL_DEVICE_FUNC(                                                          \
      int start_atom1,                                                       \
      int start_atom2,                                                       \
      int atom_tile_ind1,                                                    \
      int atom_tile_ind2,                                                    \
      LJLKScoringData<Real> const &intra_dat)                                \
      ->std::array<Real, 2> {                                                \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;                      \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;                      \
    int const separation = block_type_path_distance[intra_dat.r1.block_type] \
                                                   [atom_ind1][atom_ind2];   \
    return atom_pair_func(                                                   \
        atom_tile_ind1,                                                      \
        atom_tile_ind2,                                                      \
        start_atom1,                                                         \
        start_atom2,                                                         \
        intra_dat,                                                           \
        separation);                                                         \
  }

// SCORE_INTER_LK_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->Real
// captures:
//    None
#define SCORE_INTER_LK_ATOM_PAIR(atom_pair_func)                              \
  TMOL_DEVICE_FUNC(                                                           \
      int start_atom1,                                                        \
      int start_atom2,                                                        \
      int atom_heavy_tile_ind1,                                               \
      int atom_heavy_tile_ind2,                                               \
      LJLKScoringData<Real> const &inter_dat)                                 \
      ->std::array<Real, 1> {                                                 \
    int const atom_tile_ind1 = inter_dat.r1.heavy_inds[atom_heavy_tile_ind1]; \
    int const atom_tile_ind2 = inter_dat.r2.heavy_inds[atom_heavy_tile_ind2]; \
    int separation = interres_count_pair_separation<TILE_SIZE>(               \
        inter_dat, atom_tile_ind1, atom_tile_ind2);                           \
    Real lk = atom_pair_func(                                                 \
        atom_tile_ind1,                                                       \
        atom_tile_ind2,                                                       \
        start_atom1,                                                          \
        start_atom2,                                                          \
        inter_dat,                                                            \
        separation);                                                          \
    return {lk};                                                              \
  }

// SCORE_INTRA_LK_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->Real
// captures:
//    block_type_path_distance
#define SCORE_INTRA_LK_ATOM_PAIR(atom_pair_func)                              \
  TMOL_DEVICE_FUNC(                                                           \
      int start_atom1,                                                        \
      int start_atom2,                                                        \
      int atom_heavy_tile_ind1,                                               \
      int atom_heavy_tile_ind2,                                               \
      LJLKScoringData<Real> const &intra_dat)                                 \
      ->std::array<Real, 1> {                                                 \
    int const atom_tile_ind1 = intra_dat.r1.heavy_inds[atom_heavy_tile_ind1]; \
    int const atom_tile_ind2 = intra_dat.r2.heavy_inds[atom_heavy_tile_ind2]; \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;                       \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;                       \
    int const separation = block_type_path_distance[intra_dat.r1.block_type]  \
                                                   [atom_ind1][atom_ind2];    \
    Real lk = atom_pair_func(                                                 \
        atom_tile_ind1,                                                       \
        atom_tile_ind2,                                                       \
        start_atom1,                                                          \
        start_atom2,                                                          \
        intra_dat,                                                            \
        separation);                                                          \
    return {lk};                                                              \
  }

// SCORE_INTRA_LK_ATOM_PAIR
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                       \
      int pose_ind,                                                       \
      LJLKSingleResData<Real> &r_dat,                                     \
      int n_atoms_to_load,                                                \
      int start_atom) {                                                   \
    ljlk_load_block_coords_and_params_into_shared<DeviceDispatch, D, nt>( \
        coords,                                                           \
        block_type_atom_types,                                            \
        type_params,                                                      \
        block_type_heavy_atoms_in_tile,                                   \
        pose_ind,                                                         \
        r_dat,                                                            \
        n_atoms_to_load,                                                  \
        start_atom);                                                      \
  }

// LOAD_BLOCK_INTO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_path_distance (TView<Int, 3, D>)
#define LOAD_BLOCK_INTO_SHARED                                     \
  TMOL_DEVICE_FUNC(                                                \
      int pose_ind,                                                \
      LJLKSingleResData<Real> &r_dat,                              \
      int n_atoms_to_load,                                         \
      int start_atom,                                              \
      bool count_pair_striking_dist,                               \
      unsigned char *__restrict__ conn_ats) {                      \
    ljlk_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE>( \
        coords,                                                    \
        block_type_atom_types,                                     \
        type_params,                                               \
        block_type_heavy_atoms_in_tile,                            \
        block_type_path_distance,                                  \
        pose_ind,                                                  \
        r_dat,                                                     \
        n_atoms_to_load,                                           \
        start_atom,                                                \
        count_pair_striking_dist,                                  \
        conn_ats);                                                 \
  }

// LOAD_TILE_INVARIANT_INTERRES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Vec<Real, 3>, 2, D>)
//    pose_stack_min_bond_separation (TView<Int, 3, D>)
//    block_type_n_interblock_bonds (TView<Int, 1, D>)
//    block_type_atoms_forming_chemical_bonds (TView<Int, 2, D>)
//    pose_stack_inter_block_bondsep (TView<Int, 5, D>)
//    global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTERRES_DATA                          \
  TMOL_DEVICE_FUNC(                                                \
      int pose_ind,                                                \
      int block_ind1,                                              \
      int block_ind2,                                              \
      int block_type1,                                             \
      int block_type2,                                             \
      int n_atoms1,                                                \
      int n_atoms2,                                                \
      LJLKScoringData<Real> &inter_dat,                            \
      shared_mem_union &shared) {                                  \
    ljlk_load_tile_invariant_interres_data<DeviceDispatch, D, nt>( \
        pose_stack_block_coord_offset,                             \
        pose_stack_min_bond_separation,                            \
        block_type_n_interblock_bonds,                             \
        block_type_atoms_forming_chemical_bonds,                   \
        pose_stack_inter_block_bondsep,                            \
        global_params,                                             \
        max_important_bond_separation,                             \
        pose_ind,                                                  \
        block_ind1,                                                \
        block_ind2,                                                \
        block_type1,                                               \
        block_type2,                                               \
        n_atoms1,                                                  \
        n_atoms2,                                                  \
        inter_dat,                                                 \
        shared.m);                                                 \
  }

// LOAD_INTERRES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_path_distance (TView<Int, 3, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTERRES1_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom1,                                              \
      int n_atoms_to_load1,                                         \
      LJLKScoringData<Real> &inter_dat,                             \
      shared_mem_union &shared) {                                   \
    ljlk_load_interres1_tile_data_to_shared<DeviceDispatch, D, nt>( \
        coords,                                                     \
        block_type_atom_types,                                      \
        type_params,                                                \
        block_type_heavy_atoms_in_tile,                             \
        block_type_path_distance,                                   \
        block_type_n_heavy_atoms_in_tile,                           \
        tile_ind,                                                   \
        start_atom1,                                                \
        n_atoms_to_load1,                                           \
        inter_dat,                                                  \
        shared.m);                                                  \
  }

// LOAD_INTERRES2_TILE_DATA_TO_SHARED
//   same as LOAD_INTERRES1_TILE_DATA_TO_SHARED but saves to inter_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_path_distance (TView<Int, 3, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTERRES2_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom2,                                              \
      int n_atoms_to_load2,                                         \
      LJLKScoringData<Real> &inter_dat,                             \
      shared_mem_union &shared) {                                   \
    ljlk_load_interres2_tile_data_to_shared<DeviceDispatch, D, nt>( \
        coords,                                                     \
        block_type_atom_types,                                      \
        type_params,                                                \
        block_type_heavy_atoms_in_tile,                             \
        block_type_path_distance,                                   \
        block_type_n_heavy_atoms_in_tile,                           \
        tile_ind,                                                   \
        start_atom2,                                                \
        n_atoms_to_load2,                                           \
        inter_dat,                                                  \
        shared.m);                                                  \
  }

// LOAD_INTERRES_DATA_FROM_SHARED
// captures:
//    nothing
#define LOAD_INTERRES_DATA_FROM_SHARED                                        \
  TMOL_DEVICE_FUNC(                                                           \
      int, int, shared_mem_union &shared, LJLKScoringData<Real> &inter_dat) { \
    ljlk_load_interres_data_from_shared(shared.m, inter_dat);                 \
  }

// EVAL_INTERRES_ATOM_PAIR_SCORES
// captures:
//    score_inter_lj_atom_pair (lambda)
//    score_inter_lk_atom_pair (lambda)
#define EVAL_INTERRES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real> &inter_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto LJ = tmol::score::common::InterResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          2,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_interres_atom_pair(                                          \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_inter_lj_atom_pair,                                     \
              inter_dat);                                                   \
                                                                            \
      inter_dat.total_ljatr += std::get<0>(LJ);                             \
      inter_dat.total_ljrep += std::get<1>(LJ);                             \
                                                                            \
      auto LK = tmol::score::common::InterResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          HeavyAtomPairSelector,                                            \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          1,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_interres_atom_pair(                                          \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_inter_lk_atom_pair,                                     \
              inter_dat);                                                   \
      inter_dat.total_lk += std::get<0>(LK);                                \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                  \
        eval_scores_for_atom_pairs);                                        \
  }

// STORE_CALCULATED_ENERGIES
//    store energies if we are NOT computing per-blockpair
// captures:
//    output (TView<Real, 4, D>)
#define STORE_CALCULATED_ENERGIES                                           \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real> &score_dat, shared_mem_union &shared) {         \
    auto reduce_energies = ([&](int tid) {                                  \
      Real const cta_total_ljatr =                                          \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljatr, shared, mgpu::plus_t<Real>());         \
      Real const cta_total_ljrep =                                          \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljrep, shared, mgpu::plus_t<Real>());         \
      Real const cta_total_lk =                                             \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_lk, shared, mgpu::plus_t<Real>());            \
                                                                            \
      if (tid == 0) {                                                       \
        accumulate<D, Real>::add(                                           \
            output[0][score_dat.pose_ind][0][0], cta_total_ljatr);          \
        accumulate<D, Real>::add(                                           \
            output[1][score_dat.pose_ind][0][0], cta_total_ljrep);          \
        accumulate<D, Real>::add(                                           \
            output[2][score_dat.pose_ind][0][0], cta_total_lk);             \
      }                                                                     \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

// STORE_CALCULATED_ENERGIES
//    store energies if we ARE computing per-blockpair
// captures:
//    output (TView<Real, 4, D>)
#define STORE_CALCULATED_ENERGIES_BLOCKPAIR                                 \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real> &score_dat, shared_mem_union &shared) {         \
    auto reduce_energies = ([&](int tid) {                                  \
      Real const cta_total_ljatr =                                          \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljatr, shared, mgpu::plus_t<Real>());         \
      Real const cta_total_ljrep =                                          \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljrep, shared, mgpu::plus_t<Real>());         \
      Real const cta_total_lk =                                             \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_lk, shared, mgpu::plus_t<Real>());            \
                                                                            \
      if (tid == 0) {                                                       \
        if (score_dat.block_ind1 == score_dat.block_ind2) {                 \
          output[0][score_dat.pose_ind][score_dat.block_ind1]               \
                [score_dat.block_ind1] = cta_total_ljatr;                   \
          output[1][score_dat.pose_ind][score_dat.block_ind1]               \
                [score_dat.block_ind1] = cta_total_ljrep;                   \
          output[2][score_dat.pose_ind][score_dat.block_ind1]               \
                [score_dat.block_ind1] = cta_total_lk;                      \
        } else {                                                            \
          output[0][score_dat.pose_ind][score_dat.block_ind1]               \
                [score_dat.block_ind2] = 0.5 * cta_total_ljatr;             \
          output[1][score_dat.pose_ind][score_dat.block_ind1]               \
                [score_dat.block_ind2] = 0.5 * cta_total_ljrep;             \
          output[2][score_dat.pose_ind][score_dat.block_ind1]               \
                [score_dat.block_ind2] = 0.5 * cta_total_lk;                \
          output[0][score_dat.pose_ind][score_dat.block_ind2]               \
                [score_dat.block_ind1] = 0.5 * cta_total_ljatr;             \
          output[1][score_dat.pose_ind][score_dat.block_ind2]               \
                [score_dat.block_ind1] = 0.5 * cta_total_ljrep;             \
          output[2][score_dat.pose_ind][score_dat.block_ind2]               \
                [score_dat.block_ind1] = 0.5 * cta_total_lk;                \
        }                                                                   \
      }                                                                     \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

// LOAD_TILE_INVARIANT_INTRARES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Int, 2, D>)
//    global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTRARES_DATA                          \
  TMOL_DEVICE_FUNC(                                                \
      int pose_ind,                                                \
      int block_ind1,                                              \
      int block_type1,                                             \
      int n_atoms1,                                                \
      LJLKScoringData<Real> &intra_dat,                            \
      shared_mem_union &shared) {                                  \
    ljlk_load_tile_invariant_intrares_data<DeviceDispatch, D, nt>( \
        pose_stack_block_coord_offset,                             \
        global_params,                                             \
        max_important_bond_separation,                             \
        pose_ind,                                                  \
        block_ind1,                                                \
        block_type1,                                               \
        n_atoms1,                                                  \
        intra_dat,                                                 \
        shared.m);                                                 \
  }

// LOAD_INTRARES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTRARES1_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom1,                                              \
      int n_atoms_to_load1,                                         \
      LJLKScoringData<Real> &intra_dat,                             \
      shared_mem_union &shared) {                                   \
    ljlk_load_intrares1_tile_data_to_shared<DeviceDispatch, D, nt>( \
        coords,                                                     \
        block_type_atom_types,                                      \
        type_params,                                                \
        block_type_n_heavy_atoms_in_tile,                           \
        block_type_heavy_atoms_in_tile,                             \
        tile_ind,                                                   \
        start_atom1,                                                \
        n_atoms_to_load1,                                           \
        intra_dat,                                                  \
        shared.m);                                                  \
  }

// LOAD_INTRARES2_TILE_DATA_TO_SHARED
//    same as LOAD_INTRARES1_TILE_DATA_TO_SHARED but assign to intra_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_n_heavy_atoms_in_tile (TView<Int, 2, D>)
//    block_type_heavy_atoms_in_tile (TView<Int, 2, D>)
#define LOAD_INTRARES2_TILE_DATA_TO_SHARED                          \
  TMOL_DEVICE_FUNC(                                                 \
      int tile_ind,                                                 \
      int start_atom2,                                              \
      int n_atoms_to_load2,                                         \
      LJLKScoringData<Real> &intra_dat,                             \
      shared_mem_union &shared) {                                   \
    ljlk_load_intrares2_tile_data_to_shared<DeviceDispatch, D, nt>( \
        coords,                                                     \
        block_type_atom_types,                                      \
        type_params,                                                \
        block_type_n_heavy_atoms_in_tile,                           \
        block_type_heavy_atoms_in_tile,                             \
        tile_ind,                                                   \
        start_atom2,                                                \
        n_atoms_to_load2,                                           \
        intra_dat,                                                  \
        shared.m);                                                  \
  }

// LOAD_INTRARES_DATA_FROM_SHARED
// captures:
//     nothing
#define LOAD_INTRARES_DATA_FROM_SHARED              \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind1,                                \
      int tile_ind2,                                \
      shared_mem_union &shared,                     \
      LJLKScoringData<Real> &intra_dat) {           \
    ljlk_load_intrares_data_from_shared(            \
        tile_ind1, tile_ind2, shared.m, intra_dat); \
  }

// EVAL_INTRARES_ATOM_PAIR_SCORES
// captures:
//    score_intra_lj_atom_pair (lambda)
//    score_intra_lk_atom_pair (lambda)
#define EVAL_INTRARES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real> &intra_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto LJ = tmol::score::common::IntraResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          2,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_intrares_atom_pairs(                                         \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_intra_lj_atom_pair,                                     \
              intra_dat);                                                   \
                                                                            \
      intra_dat.total_ljatr += std::get<0>(LJ);                             \
      intra_dat.total_ljrep += std::get<1>(LJ);                             \
                                                                            \
      auto LK = tmol::score::common::IntraResBlockEvaluation<               \
          LJLKScoringData,                                                  \
          HeavyAtomPairSelector,                                            \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          1,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_intrares_atom_pairs(                                         \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_intra_lk_atom_pair,                                     \
              intra_dat);                                                   \
                                                                            \
      intra_dat.total_lk += std::get<0>(LK);                                \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                  \
        eval_scores_for_atom_pairs);                                        \
  }
// end of macro definitions

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,

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
    tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 3, D>, TPack<Int, 3, D> > {
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

  // auto output_t =
  //     TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
  TPack<Real, 4, D> output_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, D>::zeros({3, n_poses, 1, 1});
  }

  auto output = output_t.view;

  auto dV_dcoords_t =
      TPack<Vec<Real, 3>, 3, D>::zeros({3, n_poses, max_n_pose_atoms});
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

  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int,
                                int,
                                LJLKScoringData<Real> const &score_dat,
                                int cp_separation) {
      return lj_atom_energy(
          atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
    });

    auto atom_pair_lk_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int,
                                int,
                                LJLKScoringData<Real> const &score_dat,
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

    auto store_calculated_energies = ([=] STORE_CALCULATED_ENERGIES_BLOCKPAIR);

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

    tmol::score::common::tile_evaluate_block_pair<
        DeviceDispatch,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
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

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn = ([=] TMOL_DEVICE_FUNC(
                                int atom_tile_ind1,
                                int atom_tile_ind2,
                                int start_atom1,
                                int start_atom2,
                                LJLKScoringData<Real> const &score_dat,
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
                                LJLKScoringData<Real> const &score_dat,
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
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_CALCULATED_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_block_pair<
        DeviceDispatch,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
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

  if (output_block_pair_energies) {
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        n_block_pairs, eval_energies_by_block);
  } else {
    DeviceDispatch<D>::template foreach_workgroup<launch_t>(
        n_block_pairs, eval_energies);
  }

  return {output_t, dV_dcoords_t, scratch_block_neighbors_t};
}  // LJLKPoseScoreDispatch::forward

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,

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

    TView<Int, 3, D> scratch_block_neighbors,  // from forward pass
    TView<Real, 4, D> dTdV                     // nterms x nposes x len x len
    ) -> TPack<Vec<Real, 3>, 3, D> {
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

  assert(scratch_block_neighbors.size(0) == n_poses);
  assert(scratch_block_neighbors.size(1) == max_n_blocks);
  assert(scratch_block_neighbors.size(2) == max_n_blocks);

  assert(dTdV.size(0) == 3);
  assert(dTdV.size(1) == n_poses);
  assert(dTdV.size(2) == max_n_blocks);
  assert(dTdV.size(3) == max_n_blocks);

  auto dV_dcoords_t =
      TPack<Vec<Real, 3>, 3, D>::zeros({3, n_poses, max_n_pose_atoms});
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
             LJLKScoringData<Real> const &score_dat,
             int cp_separation) -> std::array<Real, 2> {
          lj_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV,       // captured
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
             LJLKScoringData<Real> const &score_dat,
             int cp_separation) -> Real {
          lk_atom_derivs(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV,       // captured
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
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies =
        ([=](LJLKScoringData<Real> &score_dat, shared_mem_union &shared) {
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

    tmol::score::common::tile_evaluate_block_pair<
        DeviceDispatch,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
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

  // Since we have the sphere overlap results from the forward pass,
  // there's only a single kernel launch here
  int const n_block_pairs = n_poses * max_n_blocks * max_n_blocks;
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      n_block_pairs, eval_derivs);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
