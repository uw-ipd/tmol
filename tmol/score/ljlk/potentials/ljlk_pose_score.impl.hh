#pragma once

#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/portable/matrix.hh>
#include <vector>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/pose_total_reduce.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/ljlk_ctx_tensor.hh>
#include <tmol/score/ljlk/potentials/ljlk_kernels.hh>
#include <tmol/score/ljlk/potentials/ljlk_mps_launch.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

// Operator definitions; safe for CPU compilation

#include <chrono>

// MAX_N_CONN (the maximum number of inter-residue chemical bonds) and
// TILE_SIZE come from ljlk_kernels.hh, which the kernel bodies and this
// dispatch must agree on.

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using tmol::portable::Vec;

#define SCORE_INTER_LJ_ATOM_PAIR(atom_pair_func)                            \
  TMOL_DEVICE_FUNC(                                                         \
      int start_atom1,                                                      \
      int start_atom2,                                                      \
      int atom_tile_ind1,                                                   \
      int atom_tile_ind2,                                                   \
      LJLKScoringData<Real> const& inter_dat) {                             \
    int separation = interres_count_pair_separation<TILE_SIZE>(             \
        inter_dat,                                                          \
        atom_tile_ind1,                                                     \
        atom_tile_ind2,                                                     \
        block_type_all_atoms_ligand_typed[inter_dat.r1.block_type]          \
            && block_type_all_atoms_ligand_typed[inter_dat.r2.block_type]); \
    return atom_pair_func(                                                  \
        atom_tile_ind1,                                                     \
        atom_tile_ind2,                                                     \
        start_atom1,                                                        \
        start_atom2,                                                        \
        inter_dat,                                                          \
        separation);                                                        \
  }

// SCORE_INTRA_LJ_ATOM_PAIR
// input argument:  a function with signature (
//     int atom_tile_idx1
//     int atom_tile_idx2
//     int start_atom1
//     int start_atom2
//     LJLKScoringData<Real> const &score_dat
//     int cp_separation)
//   ->tmol::portable::array<Real, 2>
// captures:
//    block_type_path_distance
#define SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_func)                             \
  TMOL_DEVICE_FUNC(                                                          \
      int start_atom1,                                                       \
      int start_atom2,                                                       \
      int atom_tile_ind1,                                                    \
      int atom_tile_ind2,                                                    \
      LJLKScoringData<Real> const& intra_dat) {                              \
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

// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
#define LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                         \
      int pose_ind,                                                         \
      LJLKSingleResData<Real>& r_dat,                                       \
      int n_atoms_to_load,                                                  \
      int start_atom) {                                                     \
    ljlk_load_block_coords_and_params_into_shared<DeviceOperations, D, nt, Real, Int>( \
        rot_coords,                                                         \
        block_type_atom_types,                                              \
        type_params,                                                        \
        pose_ind,                                                           \
        r_dat,                                                              \
        n_atoms_to_load,                                                    \
        start_atom);                                                        \
  }

// LOAD_BLOCK_INTO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_path_distance (TView<Int, 3, D>)
#define LOAD_BLOCK_INTO_SHARED                                       \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      LJLKSingleResData<Real>& r_dat,                                \
      int n_atoms_to_load,                                           \
      int start_atom,                                                \
      bool count_pair_striking_dist,                                 \
      unsigned char* __restrict__ conn_ats) {                        \
    ljlk_load_block_into_shared<DeviceOperations, D, nt, TILE_SIZE, Real, Int>( \
        rot_coords,                                                  \
        block_type_atom_types,                                       \
        type_params,                                                 \
        block_type_path_distance,                                    \
        pose_ind,                                                    \
        r_dat,                                                       \
        n_atoms_to_load,                                             \
        start_atom,                                                  \
        count_pair_striking_dist,                                    \
        conn_ats);                                                   \
  }

// LOAD_TILE_INVARIANT_INTERRES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Vec<Real, 3>, 2, D>)
//    pose_stack_near_blocks (TView<Int, 4, D>)
//    block_type_n_interblock_bonds (TView<Int, 1, D>)
//    block_type_atoms_forming_chemical_bonds (TView<Int, 2, D>)
//    pose_stack_inter_block_bondsep (TView<int8_t, 5, D>)
//    global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTERRES_DATA                            \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      int rot_ind1,                                                  \
      int rot_ind2,                                                  \
      int block_ind1,                                                \
      int block_ind2,                                                \
      int block_type1,                                               \
      int block_type2,                                               \
      int n_atoms1,                                                  \
      int n_atoms2,                                                  \
      LJLKScoringData<Real>& inter_dat,                              \
      shared_mem_union& shared) {                                    \
    ljlk_load_tile_invariant_interres_data<DeviceOperations, D, nt, Int, Real>( \
        rot_coord_offset,                                            \
        pose_stack_near_blocks,                                      \
        block_type_n_interblock_bonds,                               \
        block_type_atoms_forming_chemical_bonds,                     \
        pose_stack_inter_block_bondsep,                              \
        global_params,                                               \
        max_important_bond_separation,                               \
        pose_ind,                                                    \
        rot_ind1,                                                    \
        rot_ind2,                                                    \
        block_ind1,                                                  \
        block_ind2,                                                  \
        block_type1,                                                 \
        block_type2,                                                 \
        n_atoms1,                                                    \
        n_atoms2,                                                    \
        inter_dat,                                                   \
        shared.m);                                                   \
  }

// LOAD_INTERRES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_path_distance (TView<Int, 3, D>)
#define LOAD_INTERRES1_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom1,                                                \
      int n_atoms_to_load1,                                           \
      LJLKScoringData<Real>& inter_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_interres1_tile_data_to_shared<DeviceOperations, D, nt, TILE_SIZE, MAX_N_CONN, Real, Int>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        block_type_path_distance,                                     \
        tile_ind,                                                     \
        start_atom1,                                                  \
        n_atoms_to_load1,                                             \
        inter_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTERRES2_TILE_DATA_TO_SHARED
//   same as LOAD_INTERRES1_TILE_DATA_TO_SHARED but saves to inter_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
//    block_type_path_distance (TView<Int, 3, D>)
#define LOAD_INTERRES2_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom2,                                                \
      int n_atoms_to_load2,                                           \
      LJLKScoringData<Real>& inter_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_interres2_tile_data_to_shared<DeviceOperations, D, nt, TILE_SIZE, MAX_N_CONN, Real, Int>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        block_type_path_distance,                                     \
        tile_ind,                                                     \
        start_atom2,                                                  \
        n_atoms_to_load2,                                             \
        inter_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTERRES_DATA_FROM_SHARED
// captures:
//    nothing
#define LOAD_INTERRES_DATA_FROM_SHARED \
  TMOL_DEVICE_FUNC(int, int, shared_mem_union&, LJLKScoringData<Real>&) {}

// Fused LJ/LK traversal: LJ is evaluated for every atom pair and LK for the
// heavy subset inside the pair function, reusing geometry and count-pair work.
#define EVAL_INTERRES_ATOM_PAIR_SCORES_FUSED                                \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real>& inter_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto LJLK = tmol::score::common::InterResBlockEvaluation<             \
          LJLKScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          3,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_interres_atom_pair(                                          \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_inter_ljlk_atom_pair,                                   \
              inter_dat);                                                   \
      inter_dat.total_ljatr += tmol::portable::get<0>(LJLK);                           \
      inter_dat.total_ljrep += tmol::portable::get<1>(LJLK);                           \
      inter_dat.total_lk += tmol::portable::get<2>(LJLK);                              \
    });                                                                     \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(                \
        eval_scores_for_atom_pairs);                                        \
  }

// STORE_CALCULATED_ENERGIES
//    store energies if we are NOT computing per-blockpair
// captures:
//    output_block_pair_energies (bool)
//    output (TView<Real, 4, D>)
#define STORE_POSE_CALCULATED_ENERGIES                                        \
  TMOL_DEVICE_FUNC(                                                           \
      LJLKScoringData<Real>& score_dat, shared_mem_union& shared) {           \
    auto reduce_energies = ([&](int tid) {                                    \
      (void)cpu_pose_accum;  /* nvcc: first-capture outside the */   \
      /* constexpr-if below, which may not introduce a capture */   \
      Real const cta_total_ljatr =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljatr, shared, tmol::portable::plus_t<Real>()); \
      Real const cta_total_ljrep =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljrep, shared, tmol::portable::plus_t<Real>()); \
      Real const cta_total_lk =                                               \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_lk, shared, tmol::portable::plus_t<Real>());    \
                                                                              \
      if (tid == 0) {                                                         \
        int const p = score_dat.pose_ind;                                     \
        if (output_block_pair_energies) {                                     \
          int const b1 = score_dat.block_ind1;                                \
          int const b2 = score_dat.block_ind2;                                \
          output[0][p][b1][b2] = cta_total_ljatr;                             \
          output[1][p][b1][b2] = cta_total_ljrep;                             \
          output[2][p][b1][b2] = cta_total_lk;                                \
        } else {                                                              \
          if constexpr (D == tmol::Device::CPU) {                             \
            cpu_pose_accum[3 * p] += cta_total_ljatr;                         \
            cpu_pose_accum[3 * p + 1] += cta_total_ljrep;                     \
            cpu_pose_accum[3 * p + 2] += cta_total_lk;                        \
          } else {                                                            \
            accumulate<D, Real>::add(output[0][p][0][0], cta_total_ljatr);    \
            accumulate<D, Real>::add(output[1][p][0][0], cta_total_ljrep);    \
            accumulate<D, Real>::add(output[2][p][0][0], cta_total_lk);       \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    });                                                                       \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

// STORE_ROTAMER_CALCULATED_ENERGIES
//    store energies if we ARE computing rotamer pair energies
// captures:
//    cta (int)
//    output (TView<Real, 4, D>)
#define STORE_ROTAMER_CALCULATED_ENERGIES                                     \
  TMOL_DEVICE_FUNC(                                                           \
      LJLKScoringData<Real>& score_dat, shared_mem_union& shared) {           \
    auto reduce_energies = ([&](int tid) {                                    \
      Real const cta_total_ljatr =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljatr, shared, tmol::portable::plus_t<Real>()); \
      Real const cta_total_ljrep =                                            \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_ljrep, shared, tmol::portable::plus_t<Real>()); \
      Real const cta_total_lk =                                               \
          DeviceOperations<D>::template reduce_in_workgroup<nt>(              \
              score_dat.total_lk, shared, tmol::portable::plus_t<Real>());    \
      if (tid == 0) {                                                         \
        output[0][cta] = cta_total_ljatr;                                     \
        output[1][cta] = cta_total_ljrep;                                     \
        output[2][cta] = cta_total_lk;                                        \
      }                                                                       \
    });                                                                       \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(reduce_energies); \
  }

// LOAD_TILE_INVARIANT_INTRARES_DATA
// captures:
//    pose_stack_block_coord_offset (TView<Int, 2, D>)
//    global_params (TView<LJGlobalParams<Real>, 1, D>)
//    max_important_bond_separation (int)
#define LOAD_TILE_INVARIANT_INTRARES_DATA                            \
  TMOL_DEVICE_FUNC(                                                  \
      int pose_ind,                                                  \
      int rot_ind1,                                                  \
      int block_ind1,                                                \
      int block_type1,                                               \
      int n_atoms1,                                                  \
      LJLKScoringData<Real>& intra_dat,                              \
      shared_mem_union& shared) {                                    \
    ljlk_load_tile_invariant_intrares_data<DeviceOperations, D, nt, Int, Real>( \
        rot_coord_offset,                                            \
        global_params,                                               \
        max_important_bond_separation,                               \
        pose_ind,                                                    \
        rot_ind1,                                                    \
        block_ind1,                                                  \
        block_type1,                                                 \
        n_atoms1,                                                    \
        intra_dat,                                                   \
        shared.m);                                                   \
  }

// LOAD_INTRARES1_TILE_DATA_TO_SHARED
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
#define LOAD_INTRARES1_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom1,                                                \
      int n_atoms_to_load1,                                           \
      LJLKScoringData<Real>& intra_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_intrares1_tile_data_to_shared<DeviceOperations, D, nt, TILE_SIZE, MAX_N_CONN, Real, Int>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        tile_ind,                                                     \
        start_atom1,                                                  \
        n_atoms_to_load1,                                             \
        intra_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTRARES2_TILE_DATA_TO_SHARED
//    same as LOAD_INTRARES1_TILE_DATA_TO_SHARED but assign to intra_dat.r2
// captures:
//    coords (TView<Vec<Real, 3>, 2, D>)
//    block_type_atom_types (TView<Int, 2, D>)
//    type_params (TView<LJLKTypeParams<Real>, 1, D>)
#define LOAD_INTRARES2_TILE_DATA_TO_SHARED                            \
  TMOL_DEVICE_FUNC(                                                   \
      int tile_ind,                                                   \
      int start_atom2,                                                \
      int n_atoms_to_load2,                                           \
      LJLKScoringData<Real>& intra_dat,                               \
      shared_mem_union& shared) {                                     \
    ljlk_load_intrares2_tile_data_to_shared<DeviceOperations, D, nt, TILE_SIZE, MAX_N_CONN, Real, Int>( \
        rot_coords,                                                   \
        block_type_atom_types,                                        \
        type_params,                                                  \
        tile_ind,                                                     \
        start_atom2,                                                  \
        n_atoms_to_load2,                                             \
        intra_dat,                                                    \
        shared.m);                                                    \
  }

// LOAD_INTRARES_DATA_FROM_SHARED
// captures:
//     nothing
#define LOAD_INTRARES_DATA_FROM_SHARED              \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind1,                                \
      int tile_ind2,                                \
      shared_mem_union& shared,                     \
      LJLKScoringData<Real>& intra_dat) {           \
    ljlk_load_intrares_data_from_shared(            \
        tile_ind1, tile_ind2, shared.m, intra_dat); \
  }

#define EVAL_INTRARES_ATOM_PAIR_SCORES_FUSED                                \
  TMOL_DEVICE_FUNC(                                                         \
      LJLKScoringData<Real>& intra_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto LJLK = tmol::score::common::IntraResBlockEvaluation<             \
          LJLKScoringData,                                                  \
          AllAtomPairSelector,                                              \
          D,                                                                \
          TILE_SIZE,                                                        \
          nt,                                                               \
          3,                                                                \
          Real,                                                             \
          Int>::                                                            \
          eval_intrares_atom_pairs(                                         \
              tid,                                                          \
              start_atom1,                                                  \
              start_atom2,                                                  \
              score_intra_ljlk_atom_pair,                                   \
              intra_dat);                                                   \
      intra_dat.total_ljatr += tmol::portable::get<0>(LJLK);                           \
      intra_dat.total_ljrep += tmol::portable::get<1>(LJLK);                           \
      intra_dat.total_lk += tmol::portable::get<2>(LJLK);                              \
    });                                                                     \
    DeviceOperations<D>::template for_each_in_workgroup<nt>(                \
        eval_scores_for_atom_pairs);                                        \
  }
// end of macro definitions

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::
    build_compact_block_neighbors(
        ContextManager& mgr,
        TView<Vec<Real, 3>, 1, D> rot_coords,
        TView<Int, 1, D> rot_coord_offset,
        TView<Int, 2, D> first_rot_for_block,
        TView<Int, 1, D> block_ind_for_rot,
        TView<Int, 1, D> pose_ind_for_rot,
        TView<Int, 1, D> block_type_ind_for_rot,
        TView<Int, 1, D> block_type_n_atoms,
        Real reach) -> TPack<Int, 1, D> {
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const n_pairs = score::common::checked_triangular_size(
      max_n_blocks, true, "LJ/LK compact block-neighbor dispatch");
  int const n_candidates = score::common::checked_dispatch_product(
      n_poses, n_pairs, "LJ/LK compact block-neighbor dispatch");
  auto block_spheres_t =
      D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                       : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
  if constexpr (D == Device::CUDA) {
    auto neighbor_indices_t =
        rot_coord_offset.size(0) == 0
            ? TPack<Int, 1, D>::zeros({int64_t(n_candidates) + 1})
            : TPack<Int, 1, D>::empty({int64_t(n_candidates) + 1});
    score::common::sphere_overlap::
        compute_block_spheres<DeviceOperations, D, Real, Int, true>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            block_spheres_t.view,
            neighbor_indices_t.view.data());
    score::common::sphere_overlap::
        detect_compact_block_neighbors<DeviceOperations, D, Real, Int>::f(
            mgr,
            first_rot_for_block,
            block_spheres_t.view,
            neighbor_indices_t.view,
            reach);
    return neighbor_indices_t;
  } else if constexpr (D == Device::MPS) {
    // The compact block-neighbor list is a CPU/CUDA optimization; the Metal
    // forward tests for MPS before reaching the paths that consume it, so
    // this is unreachable rather than merely unported. The CPU branch below
    // cannot be instantiated here in any case -- its spatial-sweep helper
    // reads host memory, and an MPS tensor's data pointer is an MTLBuffer.
    throw std::runtime_error(
        "tmol: LJ/LK compact block neighbors are not available on the Metal "
        "backend");
  } else {
    score::common::sphere_overlap::
        compute_block_spheres<DeviceOperations, D, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            block_spheres_t.view);
    std::vector<Int> retained;
    if (score::common::sphere_overlap::try_cpu_spatial_compact_block_neighbors(
            first_rot_for_block, block_spheres_t.view, reach, retained)) {
      auto neighbor_indices_t =
          TPack<Int, 1, D>::empty({int64_t(retained.size()) + 1});
      neighbor_indices_t.view[0] = static_cast<Int>(retained.size());
      if (!retained.empty()) {
        std::memcpy(
            neighbor_indices_t.view.data() + 1,
            retained.data(),
            retained.size() * sizeof(Int));
      }
      return neighbor_indices_t;
    }
    auto neighbor_indices_t =
        TPack<Int, 1, D>::zeros({int64_t(n_candidates) + 1});
    score::common::sphere_overlap::
        detect_compact_block_neighbors<DeviceOperations, D, Real, Int>::f(
            mgr,
            first_rot_for_block,
            block_spheres_t.view,
            neighbor_indices_t.view,
            reach);
    return neighbor_indices_t;
  }
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::forward(
    ContextManager& mgr,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,

    Real max_dis,

    TView<Int, 1, D> shared_compact_block_neighbors,

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

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
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

  // Block-pair autograd recomputes coordinate derivatives in backward.
  bool const accumulate_derivs =
      require_gradient && !output_block_pair_energies;
  auto dV_dcoords_t = accumulate_derivs
                          ? TPack<Vec<Real, 3>, 2, D>::zeros({3, n_atoms})
                          : TPack<Vec<Real, 3>, 2, D>::empty({3, 0});
  auto dV_dcoords = dV_dcoords_t.view;

  bool const use_shared_compact_block_neighbors =
      shared_compact_block_neighbors.size(0) != 0;
  TPack<Real, 3, D> scratch_rot_spheres_t;
  TPack<Int, 3, D> scratch_rot_neighbors_t;
  TView<Real, 3, D> scratch_rot_spheres;
  TView<Int, 3, D> scratch_rot_neighbors;
  if (!use_shared_compact_block_neighbors) {
    scratch_rot_spheres_t =
        D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                         : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
    scratch_rot_neighbors_t =
        D == Device::CPU
            ? TPack<Int, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks})
            : TPack<Int, 3, D>::empty({n_poses, max_n_blocks, max_n_blocks});
    scratch_rot_spheres = scratch_rot_spheres_t.view;
    scratch_rot_neighbors = scratch_rot_neighbors_t.view;
  }

  // Metal accumulates whole-pose totals into per-block-pair cells and reduces
  // them afterwards; see the same comment in elec_pose_score.impl.hh for why.
  // In short: one atomic per threadgroup is already the CUDA shape, but
  // thousands of them per pose still land in a varying order, float addition
  // is not associative, and a line search that finite-differences the energy
  // cannot tolerate a non-reproducible total. Writing a cell per threadgroup
  // removes the atomic; summing the cells is fixed-order. The CPU path below
  // already avoids this with a serial double accumulator.
  bool const mps_cell_reduction =
      D == tmol::Device::MPS && !output_block_pair_energies;
  TPack<Real, 4, D> output_t;
  TPack<Real, 4, D> cells_t;
  if (output_block_pair_energies) {
    output_t =
        TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
  } else {
    output_t = TPack<Real, 4, D>::zeros({3, n_poses, 1, 1});
    if (mps_cell_reduction) {
      cells_t =
          TPack<Real, 4, D>::zeros({3, n_poses, max_n_blocks, max_n_blocks});
    }
  }
  auto output = mps_cell_reduction ? cells_t.view : output_t.view;

  // CPU workgroups execute serially. Accumulate their totals in double so
  // thousands of small block-pair terms are not rounded away in a large
  // float pose total. Atom-pair math, derivatives and returned dtype stay Real.
  // The CUDA path does not allocate or access this host scratch storage.
  std::vector<double> cpu_pose_totals;
  if constexpr (D == tmol::Device::CPU) {
    if (!output_block_pair_energies) cpu_pose_totals.resize(3 * n_poses, 0.0);
  }
  double* cpu_pose_accum = cpu_pose_totals.data();

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Large workloads benefit from a second kernel variant whose register budget
  // exposes a full 32 resident warps per SM. Keep the unconstrained variant for
  // small workloads, where register pressure is less important than avoiding
  // spills.
  LAUNCH_BOX_32_OCC_AS(launch_t_high_occupancy, 32);
  constexpr int min_high_occupancy_workgroups = 1 << 14;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;
  // The total number of unique block pairs (including self-pairs)
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "LJ/LK block-pair dispatch");

  // There are two versions of scoring:
  // Block-pair scoring, where the output is written to an n-pose x n-blocks x
  // n-blocks tensor, and whole-pose scoring, where the output is
  // atomic-incremented into an n-pose x 1 x 1 tensor. In the whole-pose scoring
  // case, we compute the atomic derivatives in the forward pass and then weight
  // them by whatever weights are applied to the terms; in a very minimal
  // backward pass (see compiled_ops.cpp). Thread 0 in each CTA atomic- adds the
  // block-pair energy for the block pair it is assigned to. In the block-pair
  // scoring case, we are unable to compute the atomic derivatives in the
  // forward pass, because how the block-pair outputs will be weighted is not
  // known until the backward pass and the memory to store atom-pair derivatives
  // for each interacting block pair would be prohibitive. So we do not compute
  // derivatives at all in the forward pass. The same energy-only kernel also
  // handles whole-pose calls that do not require gradients, avoiding the
  // register footprint and runtime branches of the derivative-capable kernel.
  // State the kernel bodies read, named explicitly rather than captured
  // implicitly by [=]. The two lambdas below close over this one object, and
  // the Metal launch binds the same fields positionally; see ljlk_ctx.hh.
  LJLKCtx<Real, Int, D> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.pose_ind_for_rot = pose_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.first_rot_block_type = first_rot_block_type;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.scratch_rot_neighbors = scratch_rot_neighbors;
  ctx.shared_compact_block_neighbors = shared_compact_block_neighbors;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_atom_types = block_type_atom_types;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.type_params = type_params;
  ctx.global_params = global_params;
  ctx.output = output;
  ctx.dV_dcoords = dV_dcoords;
  ctx.max_n_blocks = max_n_blocks;
  ctx.max_n_upper_triangle_inds = max_n_upper_triangle_inds;
  // The store writes cells whenever the output view is cell-shaped.
  ctx.output_block_pair_energies =
      output_block_pair_energies || mps_cell_reduction;
  ctx.use_shared_compact_block_neighbors = use_shared_compact_block_neighbors;
  ctx.require_gradient = require_gradient;
  ctx.max_dis = max_dis;

  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef LJLKSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    ljlk_eval_energies_by_block<DeviceOperations, D, Real, Int, launch_t>(
        ctx, shared, cta, cpu_pose_accum);
  });

  auto eval_energies = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef LJLKSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    ljlk_eval_energies<DeviceOperations, D, Real, Int, launch_t>(
        ctx, shared, cta, cpu_pose_accum);
  });

  ///////////////////////////////////////////////////////////////////////

  // Three steps
  // 0: setup
  // 1: launch a kernel to find a small bounding sphere surrounding the blocks
  // 2: launch a kernel to look for spheres that are within striking distance of
  // each other
  // 3: launch a kernel to evaluate lj/lk between pairs of blocks
  // within striking distance
  if (!use_shared_compact_block_neighbors) {
    score::common::sphere_overlap::
        compute_block_spheres<DeviceOperations, D, Real, Int>::f(
            mgr,
            rot_coords,
            rot_coord_offset,
            block_ind_for_rot,
            pose_ind_for_rot,
            block_type_ind_for_rot,
            block_type_n_atoms,
            scratch_rot_spheres);

    score::common::sphere_overlap::
        detect_block_neighbors<DeviceOperations, D, Real, Int>::f(
            mgr,
            first_rot_block_type,
            scratch_rot_spheres,
            scratch_rot_neighbors,
            max_dis);
  }

  if (D == tmol::Device::MPS) {
    // Metal cannot launch a device lambda -- there is nowhere for a closure
    // to live, and entry points are compiled by a separate toolchain -- so
    // the same kernel body is reached through a named entry rather than
    // through eval_energies_by_block / eval_energies. The context those
    // lambdas close over is the same object either way.
    //
    // This is tested ahead of the compact-block-neighbor paths because those
    // launch through DeviceOperations, which on this backend would run the
    // kernel on the CPU.
    // Entry selection depends on whether gradients are wanted, not on the
    // output shape: the cell store is independent of the derivative
    // accumulation, so a whole-pose call with gradients gets both.
    ljlk_mps_launch_if<D>(
        require_gradient && !output_block_pair_energies
            ? "ljlk_eval_energies_f32"
            : "ljlk_eval_energies_by_block_f32",
        ctx,
        int64_t(n_poses) * int64_t(max_n_upper_triangle_inds));
    if (mps_cell_reduction) {
      // Compensated, fixed-order reduction rather than torch's tree sum.
      // A tree sum is reproducible but still lets the total move by several
      // ULP when a coordinate moves by a fraction of one, because its error
      // term moves with its inputs; see pose_total_reduce_kernels.hh.
      score::common::pose_total_reduce<DeviceOperations, D, Real>::f(
          mgr, cells_t.view, output_t.view);
    }
  } else if (!output_block_pair_energies && use_shared_compact_block_neighbors) {
    if (require_gradient) {
      score::common::sphere_overlap::launch_precomputed_block_neighbors<
          DeviceOperations,
          D,
          launch_t_high_occupancy,
          Int>(
          mgr,
          shared_compact_block_neighbors,
          n_poses,
          max_n_blocks,
          eval_energies);
    } else {
      score::common::sphere_overlap::launch_precomputed_block_neighbors<
          DeviceOperations,
          D,
          launch_t_high_occupancy,
          Int>(
          mgr,
          shared_compact_block_neighbors,
          n_poses,
          max_n_blocks,
          eval_energies_by_block);
    }
  } else if (output_block_pair_energies) {
    DeviceOperations<D>::template foreach_pose_workgroup<launch_t>(
        mgr, n_poses, max_n_upper_triangle_inds, eval_energies_by_block);
  } else {
    int const n_workgroups = score::common::checked_dispatch_product(
        n_poses, max_n_upper_triangle_inds, "LJ/LK block-pair dispatch");
#ifdef __NVCC__
    if (score::common::sphere_overlap::should_compact_block_neighbors(
            n_poses, max_n_blocks, require_gradient)) {
      if (require_gradient) {
        score::common::sphere_overlap::launch_compact_block_neighbors<
            DeviceOperations,
            D,
            launch_t_high_occupancy,
            Int>(mgr, scratch_rot_neighbors, eval_energies);
      } else {
        score::common::sphere_overlap::launch_compact_block_neighbors<
            DeviceOperations,
            D,
            launch_t_high_occupancy,
            Int>(mgr, scratch_rot_neighbors, eval_energies_by_block);
      }
    } else
#endif
    {
      if (!require_gradient && n_workgroups >= min_high_occupancy_workgroups) {
        DeviceOperations<D>::template foreach_pose_workgroup<
            launch_t_high_occupancy>(
            mgr, n_poses, max_n_upper_triangle_inds, eval_energies_by_block);
      } else if (!require_gradient) {
        DeviceOperations<D>::template foreach_pose_workgroup<launch_t>(
            mgr, n_poses, max_n_upper_triangle_inds, eval_energies_by_block);
      } else if (n_workgroups >= min_high_occupancy_workgroups) {
        DeviceOperations<D>::template foreach_pose_workgroup<
            launch_t_high_occupancy>(
            mgr, n_poses, max_n_upper_triangle_inds, eval_energies);
      } else {
        DeviceOperations<D>::template foreach_pose_workgroup<launch_t>(
            mgr, n_poses, max_n_upper_triangle_inds, eval_energies);
      }
    }
  }

  if constexpr (D == tmol::Device::CPU) {
    if (!output_block_pair_energies) {
      for (int p = 0; p < n_poses; ++p) {
        for (int term = 0; term < 3; ++term) {
          output[term][p][0][0] = cpu_pose_accum[3 * p + term];
        }
      }
    }
  }

  return {output_t, dV_dcoords_t, scratch_rot_neighbors_t};
}  // LJLKPoseScoreDispatch::forward

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKPoseScoreDispatch<DeviceOperations, D, Real, Int>::backward(
    ContextManager& mgr,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
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

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
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
  int const max_n_upper_triangle_inds = score::common::checked_triangular_size(
      max_n_blocks, true, "LJ/LK derivative block-pair dispatch");

  auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_ljlk_fn =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const& score_dat,
             int cp_separation) -> tmol::portable::array<Real, 3> {
          ljlk_atom_derivs<Real, D>(
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
              dTdV[2][score_dat.pose_ind][score_dat.block_ind1]
                  [score_dat.block_ind2],
              dV_dcoords,  // captured
              type_params);
          return {0.0, 0.0, 0.0};
        });

    auto score_inter_ljlk_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_ljlk_fn));
    auto score_intra_ljlk_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_ljlk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

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

    if (dTdV[0][pose_ind][block_ind1][block_ind2] == 0
        && dTdV[1][pose_ind][block_ind1][block_ind2] == 0
        && dTdV[2][pose_ind][block_ind1][block_ind2] == 0) {
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

    auto eval_interres_atom_pair_scores =
        ([=] EVAL_INTERRES_ATOM_PAIR_SCORES_FUSED);

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

    auto eval_intrares_atom_pair_scores =
        ([=] EVAL_INTRARES_ATOM_PAIR_SCORES_FUSED);

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

  DeviceOperations<D>::template foreach_pose_workgroup<launch_t>(
      mgr, n_poses, max_n_upper_triangle_inds, eval_derivs);

  return dV_dcoords_t;
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRotamerScoreDispatch<DeviceOperations, D, Real, Int>::forward(
    ContextManager& mgr,
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
    // [n_poses, max_n_blocks]; blocks sharing an id >= 0 move in
    // lockstep, so only matching rotamer indices ever coexist
    TView<Int, 2, D> lockstep_group_for_block,
    Int max_n_rots_per_pose,

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
    //////////////////////

    // LJ parameters
    TView<LJLKTypeParams<Real>, 1, D> type_params,
    TView<LJGlobalParams<Real>, 1, D> global_params,

    Real max_dis,

    // should the output be per-pose (npose x nterms x 1 x 1)
    //   or per block-pair (npose x nterms x len x len)
    bool output_block_pair_energies,
    bool require_gradient) -> std::
    tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 2, D>, TPack<Int, 2, D> > {
  using Real3 = Vec<Real, 3>;

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

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
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

  auto dV_dcoords_t = TPack<Vec<Real, 3>, 2, D>::empty({3, 0});

  auto scratch_rot_spheres_t = D == Device::CPU
                                   ? TPack<Real, 2, D>::zeros({n_rots, 4})
                                   : TPack<Real, 2, D>::empty({n_rots, 4});
  auto scratch_rot_spheres = scratch_rot_spheres_t.view;

  auto scratch_block_spheres_t =
      D == Device::CPU ? TPack<Real, 3, D>::zeros({n_poses, max_n_blocks, 4})
                       : TPack<Real, 3, D>::empty({n_poses, max_n_blocks, 4});
  auto scratch_block_spheres = scratch_block_spheres_t.view;

  score::common::sphere_overlap::
      compute_rot_spheres<DeviceOperations, D, Real, Int>::f(
          mgr,
          rot_coords,
          rot_coord_offset,
          block_type_ind_for_rot,
          block_type_n_atoms,
          scratch_rot_spheres);

  score::common::sphere_overlap::
      compute_block_spheres_from_rot_spheres<DeviceOperations, D, Real, Int>::f(
          mgr,
          scratch_rot_spheres,
          n_rots_for_block,
          rot_offset_for_block,
          scratch_block_spheres);

  auto dispatch_indices_t =
      score::common::sphere_overlap::rot_neighbor_indices_from_block_neighbors<
          DeviceOperations,
          D,
          Real,
          Int>::
          f(mgr,
            first_rot_block_type,
            scratch_block_spheres,
            n_rots_for_block,
            rot_offset_for_block,
            scratch_rot_spheres,
            lockstep_group_for_block,
            max_dis);

  auto dispatch_indices = dispatch_indices_t.view;

  TPack<Real, 2, D> output_t;
  if (output_block_pair_energies) {
    output_t = TPack<Real, 2, D>::zeros({3, dispatch_indices.size(1)});
  } else {
    output_t = TPack<Real, 2, D>::zeros({3, n_poses});
  }
  auto output = output_t.view;

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // State the kernel body reads, named explicitly rather than captured
  // implicitly by [=], so the Metal entry point can be handed the same
  // fields; see ljlk_ctx.hh.
  LJLKCtx<Real, Int, D, 2> ctx;
  ctx.rot_coords = rot_coords;
  ctx.rot_coord_offset = rot_coord_offset;
  ctx.block_ind_for_rot = block_ind_for_rot;
  ctx.pose_ind_for_rot = pose_ind_for_rot;
  ctx.block_type_ind_for_rot = block_type_ind_for_rot;
  ctx.first_rot_block_type = first_rot_block_type;
  ctx.rot_offset_for_block = rot_offset_for_block;
  ctx.pose_stack_near_blocks = pose_stack_near_blocks;
  ctx.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep;
  ctx.dispatch_indices = dispatch_indices;
  ctx.block_type_n_atoms = block_type_n_atoms;
  ctx.block_type_atom_types = block_type_atom_types;
  ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
  ctx.block_type_atoms_forming_chemical_bonds =
      block_type_atoms_forming_chemical_bonds;
  ctx.block_type_path_distance = block_type_path_distance;
  ctx.block_type_all_atoms_ligand_typed = block_type_all_atoms_ligand_typed;
  ctx.type_params = type_params;
  ctx.global_params = global_params;
  ctx.output = output;
  ctx.max_n_blocks = max_n_blocks;
  ctx.output_block_pair_energies = true;
  ctx.max_dis = max_dis;
  // Unused by the rotamer-pair body -- it walks the dispatch list rather than
  // the block-pair upper triangle, and computes no derivatives -- but the
  // Metal launcher copies every scalar slot, so none may be left
  // uninitialized.
  ctx.max_n_upper_triangle_inds = 0;
  ctx.use_shared_compact_block_neighbors = false;
  ctx.require_gradient = false;

  // Only the whole-pose CPU path accumulates in double; the rotamer path
  // writes one cell per workgroup.
  double* cpu_pose_accum = nullptr;

  auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int cta) {
    typedef LJLKSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
        shared_mem_union;
    SHARED_MEMORY shared_mem_union shared;
    ljlk_eval_rotamer_energies_by_block<
        DeviceOperations,
        D,
        Real,
        Int,
        launch_t>(ctx, shared, cta, cpu_pose_accum);
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
  // Each pair writes its own score; backward recomputes its derivatives.
  if (D == tmol::Device::MPS) {
    // Metal has nowhere for a closure to live, so the identical body is
    // reached through a named entry point instead.
    ljlk_mps_launch_if<D>(
        "ljlk_eval_rotamer_energies_by_block_f32",
        ctx,
        dispatch_indices.size(1));
  } else {
    DeviceOperations<D>::template foreach_independent_workgroup<launch_t>(
        mgr, dispatch_indices.size(1), eval_energies_by_block);
  }

  return {output_t, dV_dcoords_t, dispatch_indices_t};
}  // LJLKRotamerScoreDispatch::forward

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
auto LJLKRotamerScoreDispatch<DeviceOperations, D, Real, Int>::backward(
    ContextManager& mgr,
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

    // InterBlockBondsep: [pose, block1, slot, (block2, min separation)]
    // and [pose, block1, slot, conn1, conn2]
    TView<Int, 4, D> pose_stack_near_blocks,
    TView<int8_t, 5, D> pose_stack_inter_block_bondsep,

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
    TView<Int, 1, D> block_type_all_atoms_ligand_typed,
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

  assert(pose_stack_near_blocks.size(0) == n_poses);
  assert(pose_stack_near_blocks.size(1) == max_n_blocks);
  assert(pose_stack_near_blocks.size(3) == 2);

  assert(pose_stack_inter_block_bondsep.size(0) == n_poses);
  assert(pose_stack_inter_block_bondsep.size(1) == max_n_blocks);
  assert(
      pose_stack_inter_block_bondsep.size(2) == pose_stack_near_blocks.size(2));
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
    auto atom_pair_ljlk_fn =
        ([=] TMOL_DEVICE_FUNC(
             int atom_tile_ind1,
             int atom_tile_ind2,
             int start_atom1,
             int start_atom2,
             LJLKScoringData<Real> const& score_dat,
             int cp_separation) -> tmol::portable::array<Real, 3> {
          ljlk_atom_derivs<Real, D>(
              atom_tile_ind1,
              atom_tile_ind2,
              start_atom1,
              start_atom2,
              score_dat,
              cp_separation,
              dTdV[0][cta],
              dTdV[1][cta],
              dTdV[2][cta],
              dV_dcoords,  // captured
              type_params);
          return {0.0, 0.0, 0.0};
        });

    auto score_inter_ljlk_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_ljlk_fn));
    auto score_intra_ljlk_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_ljlk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;

    } shared;

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

    auto eval_interres_atom_pair_scores =
        ([=] EVAL_INTERRES_ATOM_PAIR_SCORES_FUSED);

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

    auto eval_intrares_atom_pair_scores =
        ([=] EVAL_INTRARES_ATOM_PAIR_SCORES_FUSED);

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
      mgr, dispatch_indices.size(1), eval_derivs);

  return dV_dcoords_t;
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
