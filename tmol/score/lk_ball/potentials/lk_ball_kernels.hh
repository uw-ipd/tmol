#pragma once

// LK-ball kernel body.
//
// Separated from lk_ball_pose_score.impl.hh -- which holds the dispatch entry
// points and therefore reaches the torch tensor layer -- so that a Metal
// translation unit can include just the kernel. Everything named here is
// freestanding: the portable math layer, the view-generic helpers in
// lk_ball.hh, and the device-templated workgroup machinery.
//
// Note the absence of device_operations.hh. The body reaches the workgroup
// primitives through its DeviceDispatch template parameter, which is only
// resolved at instantiation, so the interface declaration -- which returns
// TPack and therefore needs torch -- is the caller's concern, not its own.
//
// Only the forward energy kernel lives here. LK-ball computes derivatives in
// a separate `backward` dispatch rather than inside the forward pass, so
// unlike elec and ljlk there is one kernel family, not two.

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/lk_ball/potentials/lk_ball.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_ctx.hh>
// After lk_ball.hh, deliberately: constants.hh defines TILE_SIZE,
// MAX_N_WATER and MAX_N_CONN as macros, and those are also the names of
// lk_ball.hh's template parameters. Including it first turns
// `template <..., int TILE_SIZE, ...>` into `template <..., int 32, ...>`.
#include <tmol/score/lk_ball/potentials/constants.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

namespace common = tmol::score::common;
using tmol::portable::Vec;
using tmol::score::common::accumulate;
using tmol::score::common::WorkgroupLane;

// Shared-memory block for one LK-ball workgroup.
//
// Hoisted to namespace scope so the kernel body can take it by reference
// rather than declare it inline. Metal permits `threadgroup` declarations
// only at kernel-entry scope, so the entry declares the storage and passes it
// down; CUDA and CPU are unaffected, since the caller declares it either way.
template <typename Real, int TILE, int MAX_WATER, int MAX_CONN, typename launch_t>
union LKBallSharedMem {
  CTA_REAL_REDUCE_T_TYPEDEF;

  // Metal forbids a constructor on a `threadgroup` variable: the compiler
  // emits a guard variable and the link fails on __cxa_guard_acquire.
#if !TMOL_PORTABLE_METAL
  LKBallSharedMem() {}
#endif
  LKBallBlockPairSharedData<Real, TILE, MAX_WATER, MAX_CONN> m;
  CTA_REAL_REDUCE_T_VARIABLE;
};

// Block-pair index for one workgroup.
//
// Moved here from lk_ball_pose_score.impl.hh so the Metal entry can reach it
// without the torch tensor layer. The CPU and CUDA dispatches call it through
// the same header.
template <common::TilePairMode Mode>
TMOL_HD common::tuple<int, int, int> lk_ball_pose_pair_indices(
    int cta, int max_n_blocks, int n_pairs_with_diagonal) {
  if constexpr (Mode == common::TilePairMode::Inter) {
    int const n_pairs = n_pairs_with_diagonal - max_n_blocks;
    auto pair = common::upper_triangle_inds_from_linear_index(
        cta % n_pairs, max_n_blocks);
    return common::make_tuple(
        cta / n_pairs, common::get<0>(pair), common::get<1>(pair));
  } else if constexpr (Mode == common::TilePairMode::Intra) {
    int const block = cta % max_n_blocks;
    return common::make_tuple(cta / max_n_blocks, block, block);
  } else {
    int const n_pairs = n_pairs_with_diagonal;
    auto pair = common::upper_triangle_inds_from_linear_index(
        cta % n_pairs, max_n_blocks + 1);
    return common::make_tuple(
        cta / n_pairs, common::get<0>(pair), common::get<1>(pair) - 1);
  }
}

// TO DO: standardize tiled inter-block count pair
template <int TILE, typename InterEnergyData>
TMOL_HD int interres_count_pair_separation(
    TMOL_THREAD InterEnergyData const& inter_dat,
    int atom_tile_ind1,
    int atom_tile_ind2) {
  int separation = inter_dat.pair_data.min_separation;
  if (separation <= inter_dat.pair_data.max_important_bond_separation) {
    separation = common::count_pair::shared_mem_inter_block_separation<TILE>(
        inter_dat.pair_data.max_important_bond_separation,
        atom_tile_ind1,
        atom_tile_ind2,
        inter_dat.r1.n_conn,
        inter_dat.r2.n_conn,
        inter_dat.r1.path_dist,
        inter_dat.r2.path_dist,
        inter_dat.pair_data.conn_seps);
  }
  return separation;
}

#define LOAD_TILE_INVARIANT_INTERRES_DATA                                    \
  TMOL_DEVICE_FUNC(                                                          \
      int pose_ind,                                                          \
      int rot_ind1,                                                          \
      int rot_ind2,                                                          \
      int block_ind1,                                                        \
      int block_ind2,                                                        \
      int block_type1,                                                       \
      int block_type2,                                                       \
      int n_atoms1,                                                          \
      int n_atoms2,                                                          \
      TMOL_THREAD LKBallScoringData<Real>& inter_dat,                        \
      TMOL_SHARED_REF(Shared) shared) {                                      \
    lk_ball_load_tile_invariant_interres_data<DeviceDispatch, Dev, nt, Int>( \
        ctx.rot_coord_offset,                                                \
        ctx.block_type_ind_for_rot,                                          \
        ctx.pose_stack_inter_residue_connections,                            \
        ctx.pose_stack_near_blocks,                                          \
        ctx.pose_stack_inter_block_bondsep,                                  \
        ctx.block_type_n_interblock_bonds,                                   \
        ctx.block_type_atoms_forming_chemical_bonds,                         \
        ctx.global_params,                                                   \
        max_important_bond_separation,                                       \
        pose_ind,                                                            \
        rot_ind1,                                                            \
        rot_ind2,                                                            \
        block_ind1,                                                          \
        block_ind2,                                                          \
        block_type1,                                                         \
        block_type2,                                                         \
        n_atoms1,                                                            \
        n_atoms2,                                                            \
        inter_dat,                                                           \
        shared.m,                                                            \
        lane);                                                               \
  }

#define LOAD_INTERRES1_TILE_DATA_TO_SHARED                                    \
  TMOL_DEVICE_FUNC(                                                           \
      int tile_ind,                                                           \
      int start_atom1,                                                        \
      int n_atoms_to_load1,                                                   \
      TMOL_THREAD LKBallScoringData<Real>& inter_dat,                         \
      TMOL_SHARED_REF(Shared) shared) {                                       \
    lk_ball_load_interres1_tile_data_to_shared<DeviceDispatch, Dev, nt, Int>( \
        ctx.rot_coords,                                                       \
        ctx.water_coords,                                                     \
        ctx.block_type_tile_n_polar_atoms,                                    \
        ctx.block_type_tile_n_occluder_atoms,                                 \
        ctx.block_type_tile_pol_occ_inds,                                     \
        ctx.block_type_tile_lk_ball_params,                                   \
        ctx.block_type_path_distance,                                         \
        tile_ind,                                                             \
        start_atom1,                                                          \
        n_atoms_to_load1,                                                     \
        inter_dat,                                                            \
        shared.m,                                                             \
        lane);                                                                \
  }

#define LOAD_INTERRES2_TILE_DATA_TO_SHARED                                    \
  TMOL_DEVICE_FUNC(                                                           \
      int tile_ind,                                                           \
      int start_atom2,                                                        \
      int n_atoms_to_load2,                                                   \
      TMOL_THREAD LKBallScoringData<Real>& inter_dat,                         \
      TMOL_SHARED_REF(Shared) shared) {                                       \
    lk_ball_load_interres2_tile_data_to_shared<DeviceDispatch, Dev, nt, Int>( \
        ctx.rot_coords,                                                       \
        ctx.water_coords,                                                     \
        ctx.block_type_tile_n_polar_atoms,                                    \
        ctx.block_type_tile_n_occluder_atoms,                                 \
        ctx.block_type_tile_pol_occ_inds,                                     \
        ctx.block_type_tile_lk_ball_params,                                   \
        ctx.block_type_path_distance,                                         \
        tile_ind,                                                             \
        start_atom2,                                                          \
        n_atoms_to_load2,                                                     \
        inter_dat,                                                            \
        shared.m,                                                             \
        lane);                                                                \
  }

#define LOAD_INTERRES_DATA_FROM_SHARED                                         \
  (int, int, TMOL_SHARED_REF(Shared), TMOL_THREAD LKBallScoringData<Real>&) {}

#define EVAL_INTERRES_ATOM_PAIR_SCORES                                                    \
  TMOL_DEVICE_FUNC(                                                                       \
      TMOL_THREAD LKBallScoringData<Real>& inter_dat, int start_atom1, int start_atom2) { \
    eval_interres_pol_occ_pair_energies<DeviceDispatch, Dev, nt>(                         \
        inter_dat,                                                                        \
        start_atom1,                                                                      \
        start_atom2,                                                                      \
        score_inter_lk_ball_atom_pair,                                                    \
        lane);                                                                            \
  }

#define STORE_POSE_CALCULATED_ENERGIES                                                  \
  TMOL_DEVICE_FUNC(                                                                     \
      TMOL_THREAD LKBallScoringData<Real>& score_dat, TMOL_SHARED_REF(Shared) shared) { \
    auto reduce_energies = ([&](int tid) {                                              \
      Real const cta_total_lk_ball_iso =                                                \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_ball_iso,                                    \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
      Real const cta_total_lk_ball =                                                    \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_ball,                                        \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
      Real const cta_total_lk_bridge =                                                  \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_bridge,                                      \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
      Real const cta_total_lk_bridge_uncpl =                                            \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_bridge_uncpl,                                \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
                                                                                        \
      if (tid == 0) {                                                                   \
        if (!ctx.output_block_pair_energies) {                                          \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_ball_iso][score_dat.pair_data.pose_ind][0][0],            \
              cta_total_lk_ball_iso);                                                   \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_ball][score_dat.pair_data.pose_ind][0][0],                \
              cta_total_lk_ball);                                                       \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_bridge][score_dat.pair_data.pose_ind][0][0],              \
              cta_total_lk_bridge);                                                     \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_bridge_uncpl][score_dat.pair_data.pose_ind][0][0],        \
              cta_total_lk_bridge_uncpl);                                               \
        } else {                                                                        \
          int const p = score_dat.pair_data.pose_ind;                                   \
          int const b1 = score_dat.r1.block_ind;                                        \
          int const b2 = score_dat.r2.block_ind;                                        \
          ctx.output[w_lk_ball_iso][p][b1][b2] = cta_total_lk_ball_iso;                 \
          ctx.output[w_lk_ball][p][b1][b2] = cta_total_lk_ball;                         \
          ctx.output[w_lk_bridge][p][b1][b2] = cta_total_lk_bridge;                     \
          ctx.output[w_lk_bridge_uncpl][p][b1][b2] = cta_total_lk_bridge_uncpl;         \
        }                                                                               \
      }                                                                                 \
    });                                                                                 \
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(                            \
        reduce_energies, lane);                                                         \
  }

#define STORE_ROTAMER_CALCULATED_ENERGIES                                                  \
  TMOL_DEVICE_FUNC(                                                                     \
      TMOL_THREAD LKBallScoringData<Real>& score_dat, TMOL_SHARED_REF(Shared) shared) { \
    auto reduce_energies = ([&](int tid) {                                              \
      Real const cta_total_lk_ball_iso =                                                \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_ball_iso,                                    \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
      Real const cta_total_lk_ball =                                                    \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_ball,                                        \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
      Real const cta_total_lk_bridge =                                                  \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_bridge,                                      \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
      Real const cta_total_lk_bridge_uncpl =                                            \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(                        \
              score_dat.pair_data.total_lk_bridge_uncpl,                                \
              shared,                                                                   \
              tmol::portable::plus_t<Real>(), lane);                                    \
                                                                                        \
      if (tid == 0) {                                                                   \
        if (!ctx.output_block_pair_energies) {                                          \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_ball_iso][score_dat.pair_data.pose_ind],                  \
              cta_total_lk_ball_iso);                                                   \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_ball][score_dat.pair_data.pose_ind],                      \
              cta_total_lk_ball);                                                       \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_bridge][score_dat.pair_data.pose_ind],                    \
              cta_total_lk_bridge);                                                     \
          accumulate<Dev, Real>::add(                                                   \
              ctx.output[w_lk_bridge_uncpl][score_dat.pair_data.pose_ind],              \
              cta_total_lk_bridge_uncpl);                                               \
        } else {                                                                        \
          ctx.output[w_lk_ball_iso][cta] = cta_total_lk_ball_iso;                      \
          ctx.output[w_lk_ball][cta] = cta_total_lk_ball;                               \
          ctx.output[w_lk_bridge][cta] = cta_total_lk_bridge;                           \
          ctx.output[w_lk_bridge_uncpl][cta] = cta_total_lk_bridge_uncpl;               \
        }                                                                               \
      }                                                                                 \
    });                                                                                 \
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(                            \
        reduce_energies, lane);                                                         \
  }

#define LOAD_TILE_INVARIANT_INTRARES_DATA                                    \
  TMOL_DEVICE_FUNC(                                                          \
      int pose_ind,                                                          \
      int rot_ind1,                                                          \
      int block_ind1,                                                        \
      int block_type1,                                                       \
      int n_atoms1,                                                          \
      TMOL_THREAD LKBallScoringData<Real>& intra_dat,                        \
      TMOL_SHARED_REF(Shared) shared) {                                      \
    lk_ball_load_tile_invariant_intrares_data<DeviceDispatch, Dev, nt, Int>( \
        ctx.rot_coord_offset,                                                \
        ctx.block_type_ind_for_rot,                                          \
        ctx.global_params,                                                   \
        max_important_bond_separation,                                       \
        pose_ind,                                                            \
        rot_ind1,                                                            \
        block_ind1,                                                          \
        block_type1,                                                         \
        n_atoms1,                                                            \
        intra_dat,                                                           \
        shared.m,                                                            \
        lane);                                                               \
  }

#define LOAD_INTRARES1_TILE_DATA_TO_SHARED                                    \
  TMOL_DEVICE_FUNC(                                                           \
      int tile_ind,                                                           \
      int start_atom1,                                                        \
      int n_atoms_to_load1,                                                   \
      TMOL_THREAD LKBallScoringData<Real>& intra_dat,                         \
      TMOL_SHARED_REF(Shared) shared) {                                       \
    lk_ball_load_intrares1_tile_data_to_shared<DeviceDispatch, Dev, nt, Int>( \
        ctx.rot_coords,                                                       \
        ctx.water_coords,                                                     \
        ctx.block_type_tile_n_polar_atoms,                                    \
        ctx.block_type_tile_n_occluder_atoms,                                 \
        ctx.block_type_tile_pol_occ_inds,                                     \
        ctx.block_type_tile_lk_ball_params,                                   \
                                                                              \
        tile_ind,                                                             \
        start_atom1,                                                          \
        n_atoms_to_load1,                                                     \
        intra_dat,                                                            \
        shared.m,                                                             \
        lane);                                                                \
  }

#define LOAD_INTRARES2_TILE_DATA_TO_SHARED                                    \
  TMOL_DEVICE_FUNC(                                                           \
      int tile_ind,                                                           \
      int start_atom2,                                                        \
      int n_atoms_to_load2,                                                   \
      TMOL_THREAD LKBallScoringData<Real>& intra_dat,                         \
      TMOL_SHARED_REF(Shared) shared) {                                       \
    lk_ball_load_intrares2_tile_data_to_shared<DeviceDispatch, Dev, nt, Int>( \
        ctx.rot_coords,                                                       \
        ctx.water_coords,                                                     \
        ctx.block_type_tile_n_polar_atoms,                                    \
        ctx.block_type_tile_n_occluder_atoms,                                 \
        ctx.block_type_tile_pol_occ_inds,                                     \
        ctx.block_type_tile_lk_ball_params,                                   \
        tile_ind,                                                             \
        start_atom2,                                                          \
        n_atoms_to_load2,                                                     \
        intra_dat,                                                            \
        shared.m,                                                             \
        lane);                                                                \
  }

#define LOAD_INTRARES_DATA_FROM_SHARED                  \
  TMOL_DEVICE_FUNC(                                     \
      int tile_ind1,                                    \
      int tile_ind2,                                    \
      TMOL_SHARED_REF(Shared) shared,                   \
      TMOL_THREAD LKBallScoringData<Real>& intra_dat) { \
    lk_ball_load_intrares_data_from_shared<Dev>(        \
        tile_ind1, tile_ind2, shared.m, intra_dat);     \
  }

#define EVAL_INTRARES_ATOM_PAIR_SCORES                                                    \
  TMOL_DEVICE_FUNC(                                                                       \
      TMOL_THREAD LKBallScoringData<Real>& intra_dat, int start_atom1, int start_atom2) { \
    eval_intrares_pol_occ_pair_energies<DeviceDispatch, Dev, nt>(                         \
        intra_dat,                                                                        \
        start_atom1,                                                                      \
        start_atom2,                                                                      \
        score_intra_lk_ball_atom_pair,                                                    \
        lane);                                                                            \
  }

// Score every polar/occluder pair in one block pair.
//
// Templated on the context and shared-memory types, so the CPU and CUDA
// builds pass TView-backed state while a `.metal` entry passes MetalView-
// backed state and runs the same code.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared,
    // The compact launcher splits a large neighbour list into an inter-only
    // and an intra-only pass over the same triangular candidate ids, so the
    // body has to be told which kind of pair to score. Defaulting both to
    // InterAndIntra keeps every other caller -- and the CPU and Metal front
    // ends, which never split -- unchanged.
    common::TilePairMode PairMode = common::TilePairMode::InterAndIntra,
    common::TilePairMode IndexMode = PairMode>
TMOL_HD void lk_ball_eval_energies_by_block(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  constexpr auto pair_mode = PairMode;
  constexpr auto index_mode = IndexMode;
    auto score_inter_lk_ball_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int pol_start,
             int occ_start,
             int pol_ind,
             int occ_ind,
             TMOL_THREAD LKBallScoringData<Real>& inter_dat,
             bool polar_first) {
          int pol_tile_ind = (polar_first ? inter_dat.r1 : inter_dat.r2)
                                 .pol_occ_tile_inds[pol_ind];
          int occ_tile_ind = (polar_first ? inter_dat.r2 : inter_dat.r1)
                                 .pol_occ_tile_inds[occ_ind];
          int separation = interres_count_pair_separation<TILE_SIZE>(
              inter_dat,
              (polar_first ? pol_tile_ind : occ_tile_ind),
              (polar_first ? occ_tile_ind : pol_tile_ind));
          lk_ball_Vt<Real> E = lk_ball_atom_energy_full<MAX_N_WATER>(
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_start,
              occ_start,
              polar_first ? inter_dat.r1 : inter_dat.r2,
              polar_first ? inter_dat.r2 : inter_dat.r1,
              inter_dat.pair_data,
              separation);
          inter_dat.pair_data.total_lk_ball_iso += E.lkball_iso;
          inter_dat.pair_data.total_lk_ball += E.lkball;
          inter_dat.pair_data.total_lk_bridge += E.lkbridge;
          inter_dat.pair_data.total_lk_bridge_uncpl += E.lkbridge_uncpl;
        });

    auto score_intra_lk_ball_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int pol_start,
             int occ_start,
             int pol_ind,
             int occ_ind,
             TMOL_THREAD LKBallScoringData<Real>& intra_dat,
             bool polar_first) {
          int pol_tile_ind = (polar_first ? intra_dat.r1 : intra_dat.r2)
                                 .pol_occ_tile_inds[pol_ind];
          int occ_tile_ind = (polar_first ? intra_dat.r2 : intra_dat.r1)
                                 .pol_occ_tile_inds[occ_ind];
          int const pol_atom_ind = pol_start + pol_tile_ind;
          int const occ_atom_ind = occ_start + occ_tile_ind;

          int const separation =
              ctx.block_type_path_distance[intra_dat.r1.block_type][pol_atom_ind]
                                      [occ_atom_ind];
          lk_ball_Vt<Real> E = lk_ball_atom_energy_full<MAX_N_WATER>(
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_start,
              occ_start,
              polar_first ? intra_dat.r1 : intra_dat.r2,
              polar_first ? intra_dat.r2 : intra_dat.r1,
              intra_dat.pair_data,
              separation);
          intra_dat.pair_data.total_lk_ball_iso += E.lkball_iso;
          intra_dat.pair_data.total_lk_ball += E.lkball;
          intra_dat.pair_data.total_lk_bridge += E.lkbridge;
          intra_dat.pair_data.total_lk_bridge_uncpl += E.lkbridge_uncpl;
        });


    int const max_important_bond_separation = 4;

    auto pair_indices = lk_ball_pose_pair_indices<index_mode>(
        cta, ctx.max_n_blocks, ctx.n_block_pairs);
    int const pose_ind = common::get<0>(pair_indices);
    int const block_ind1 = common::get<1>(pair_indices);
    int const block_ind2 = common::get<2>(pair_indices);
    if constexpr (pair_mode != index_mode) {
      if constexpr (pair_mode == common::TilePairMode::Inter) {
        if (block_ind1 == block_ind2) return;
      } else {
        if (block_ind1 != block_ind2) return;
      }
    }

    // We still kill CTAs targetting non-neighboring block pairs, though,
    // and that can be a lot
    if (!ctx.use_shared_compact_block_neighbors
        && ctx.scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }
    int const rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];

    if (rot_ind1 < 0 || rot_ind2 < 0) {
      return;
    }
    //   printf("Scoring lk ball pose %d block pair %d %d\n", pose_ind,
    //   block_ind1, block_ind2);

    int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
    int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];

    int const n_atoms1 = ctx.block_type_n_atoms[block_type1];
    int const n_atoms2 = ctx.block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared =
        ([=](int, int, TMOL_SHARED_REF(Shared), TMOL_THREAD LKBallScoringData<Real>&) {});

    auto eval_interres_atom_pair_scores =
        ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_POSE_CALCULATED_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared =
        ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores =
        ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        Dev,
        LKBallScoringData<Real>,
        LKBallScoringData<Real>,
        Real,
        TILE_SIZE,
        pair_mode>(
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
        store_calculated_energies,
        lane);
}

// Derivative evaluation, backward pass. Same tiling and the same atom-pair
// traversal as the energy kernel; each pair scatters into the two
// derivative accumulators instead of reducing into a score.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared,
    // See lk_ball_eval_energies_by_block: the backward pass splits the same
    // way, so it takes the same two modes.
    common::TilePairMode PairMode = common::TilePairMode::InterAndIntra,
    common::TilePairMode IndexMode = PairMode>
TMOL_HD void lk_ball_eval_derivs(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  constexpr auto pair_mode = PairMode;
  constexpr auto index_mode = IndexMode;
    // The upstream weights are invariant for every atom pair this workgroup
    // handles. Load them once rather than per pair.
    auto weight_indices = lk_ball_pose_pair_indices<index_mode>(
        cta, ctx.max_n_blocks, ctx.n_block_pairs);
    int const w_pose_ind = common::get<0>(weight_indices);
    int const w_block_ind1 = common::get<1>(weight_indices);
    int const w_block_ind2 = common::get<2>(weight_indices);
    tmol::portable::Vec<Real, 4> local_dTdV;
    {
      int const wb1 = ctx.block_pair_scoring ? w_block_ind1 : 0;
      int const wb2 = ctx.block_pair_scoring ? w_block_ind2 : 0;
      for (int i = 0; i < 4; ++i) {
        local_dTdV[i] = ctx.dTdV[i][w_pose_ind][wb1][wb2];
      }
    }
    if (ctx.block_pair_scoring && local_dTdV[0] == 0 && local_dTdV[1] == 0
        && local_dTdV[2] == 0 && local_dTdV[3] == 0) {
      return;
    }

    auto lk_ball_atom_derivs =
        ([=] TMOL_DEVICE_FUNC(
             int pol_ind,
             int occ_ind,
             int pol_tile_ind,
             int occ_tile_ind,
             int pol_start,
             int occ_start,
             TMOL_THREAD LKBallSingleResData<Real> const& pol_dat,
             TMOL_THREAD LKBallSingleResData<Real> const& occ_dat,
             TMOL_THREAD LKBallResPairData<Real> const& respair_dat,
             int cp_separation) {
          lk_ball_atom_derivs_full<
              TILE_SIZE,
              MAX_N_WATER,
              Real,
              Dev,
              (Dev == tmol::Device::CUDA)>(
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_start,
              occ_start,
              pol_dat,
              occ_dat,
              respair_dat,
              cp_separation,
              local_dTdV,
              ctx.dV_d_pose_coords,
              ctx.dV_d_water_coords);
        });

    auto score_inter_lk_ball_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int pol_start,
             int occ_start,
             int pol_ind,
             int occ_ind,
             TMOL_THREAD LKBallScoringData<Real>& inter_dat,
             bool polar_first) {
          int pol_tile_ind = (polar_first ? inter_dat.r1 : inter_dat.r2)
                                 .pol_occ_tile_inds[pol_ind];
          int occ_tile_ind = (polar_first ? inter_dat.r2 : inter_dat.r1)
                                 .pol_occ_tile_inds[occ_ind];
          int separation = interres_count_pair_separation<TILE_SIZE>(
              inter_dat,
              (polar_first ? pol_tile_ind : occ_tile_ind),
              (polar_first ? occ_tile_ind : pol_tile_ind));
          lk_ball_atom_derivs(
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_start,
              occ_start,
              polar_first ? inter_dat.r1 : inter_dat.r2,
              polar_first ? inter_dat.r2 : inter_dat.r1,
              inter_dat.pair_data,
              separation);
        });

    auto score_intra_lk_ball_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int pol_start,
             int occ_start,
             int pol_ind,
             int occ_ind,
             TMOL_THREAD LKBallScoringData<Real>& intra_dat,
             bool polar_first) {
          int pol_tile_ind = (polar_first ? intra_dat.r1 : intra_dat.r2)
                                 .pol_occ_tile_inds[pol_ind];
          int occ_tile_ind = (polar_first ? intra_dat.r2 : intra_dat.r1)
                                 .pol_occ_tile_inds[occ_ind];
          int const pol_atom_ind = pol_start + pol_tile_ind;
          int const occ_atom_ind = occ_start + occ_tile_ind;
          int const separation =
              ctx.block_type_path_distance[intra_dat.r1.block_type]
                                          [pol_atom_ind][occ_atom_ind];
          lk_ball_atom_derivs(
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_start,
              occ_start,
              polar_first ? intra_dat.r1 : intra_dat.r2,
              polar_first ? intra_dat.r2 : intra_dat.r1,
              intra_dat.pair_data,
              separation);
        });

    int const max_important_bond_separation = 4;

    auto pair_indices = lk_ball_pose_pair_indices<index_mode>(
        cta, ctx.max_n_blocks, ctx.n_block_pairs);
    int const pose_ind = common::get<0>(pair_indices);
    int const block_ind1 = common::get<1>(pair_indices);
    int const block_ind2 = common::get<2>(pair_indices);
    if constexpr (pair_mode != index_mode) {
      if constexpr (pair_mode == common::TilePairMode::Inter) {
        if (block_ind1 == block_ind2) return;
      } else {
        if (block_ind1 != block_ind2) return;
      }
    }

    // We still kill CTAs targetting non-neighboring block pairs, though,
    // and that can be a lot
    if (!ctx.use_shared_compact_block_neighbors
        && ctx.scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
      return;
    }
    int const rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
    int const rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];

    if (rot_ind1 < 0 || rot_ind2 < 0) {
      return;
    }
    //   printf("Scoring lk ball pose %d block pair %d %d\n", pose_ind,
    //   block_ind1, block_ind2);

    int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
    int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];

    int const n_atoms1 = ctx.block_type_n_atoms[block_type1];
    int const n_atoms2 = ctx.block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared =
        ([=](int, int, TMOL_SHARED_REF(Shared), TMOL_THREAD LKBallScoringData<Real>&) {});

    auto eval_interres_atom_pair_scores =
        ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    // Derivatives only; nothing to reduce or store.
    auto store_calculated_energies = ([=](
        TMOL_THREAD LKBallScoringData<Real>&, TMOL_SHARED_REF(Shared)) {});

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared =
        ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores =
        ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        Dev,
        LKBallScoringData<Real>,
        LKBallScoringData<Real>,
        Real,
        TILE_SIZE,
        pair_mode>(
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
        store_calculated_energies,
        lane);
}

// The rotamer-pair variant: identical atom-pair work, but the block pair
// comes from the dispatch list the sphere-overlap kernels built, and every
// workgroup writes its own output cell.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void lk_ball_eval_rotamer_energies_by_block(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
    auto score_inter_lk_ball_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int pol_start,
             int occ_start,
             int pol_ind,
             int occ_ind,
             TMOL_THREAD LKBallScoringData<Real>& inter_dat,
             bool polar_first) {
          int pol_tile_ind = (polar_first ? inter_dat.r1 : inter_dat.r2)
                                 .pol_occ_tile_inds[pol_ind];
          int occ_tile_ind = (polar_first ? inter_dat.r2 : inter_dat.r1)
                                 .pol_occ_tile_inds[occ_ind];
          int separation = interres_count_pair_separation<TILE_SIZE>(
              inter_dat,
              (polar_first ? pol_tile_ind : occ_tile_ind),
              (polar_first ? occ_tile_ind : pol_tile_ind));
          lk_ball_Vt<Real> E = lk_ball_atom_energy_full<MAX_N_WATER>(
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_start,
              occ_start,
              polar_first ? inter_dat.r1 : inter_dat.r2,
              polar_first ? inter_dat.r2 : inter_dat.r1,
              inter_dat.pair_data,
              separation);
          inter_dat.pair_data.total_lk_ball_iso += E.lkball_iso;
          inter_dat.pair_data.total_lk_ball += E.lkball;
          inter_dat.pair_data.total_lk_bridge += E.lkbridge;
          inter_dat.pair_data.total_lk_bridge_uncpl += E.lkbridge_uncpl;
        });

    auto score_intra_lk_ball_atom_pair =
        ([=] TMOL_DEVICE_FUNC(
             int pol_start,
             int occ_start,
             int pol_ind,
             int occ_ind,
             TMOL_THREAD LKBallScoringData<Real>& intra_dat,
             bool polar_first) {
          int pol_tile_ind = (polar_first ? intra_dat.r1 : intra_dat.r2)
                                 .pol_occ_tile_inds[pol_ind];
          int occ_tile_ind = (polar_first ? intra_dat.r2 : intra_dat.r1)
                                 .pol_occ_tile_inds[occ_ind];
          int const pol_atom_ind = pol_start + pol_tile_ind;
          int const occ_atom_ind = occ_start + occ_tile_ind;

          int const separation =
              ctx.block_type_path_distance[intra_dat.r1.block_type][pol_atom_ind]
                                      [occ_atom_ind];
          lk_ball_Vt<Real> E = lk_ball_atom_energy_full<MAX_N_WATER>(
              pol_ind,
              occ_ind,
              pol_tile_ind,
              occ_tile_ind,
              pol_start,
              occ_start,
              polar_first ? intra_dat.r1 : intra_dat.r2,
              polar_first ? intra_dat.r2 : intra_dat.r1,
              intra_dat.pair_data,
              separation);
          intra_dat.pair_data.total_lk_ball_iso += E.lkball_iso;
          intra_dat.pair_data.total_lk_ball += E.lkball;
          intra_dat.pair_data.total_lk_bridge += E.lkbridge;
          intra_dat.pair_data.total_lk_bridge_uncpl += E.lkbridge_uncpl;
        });


    int const max_important_bond_separation = 4;

    // The block pair comes from the dispatch list rather than the upper
    // triangle; the sphere-overlap kernels already dropped the pairs that
    // cannot interact, so there is nothing further to reject here.
    int const pose_ind = ctx.dispatch_indices[0][cta];
    int const rot_ind1 = ctx.dispatch_indices[1][cta];
    int const rot_ind2 = ctx.dispatch_indices[2][cta];

    int const block_ind1 = ctx.block_ind_for_rot[rot_ind1];
    int const block_ind2 = ctx.block_ind_for_rot[rot_ind2];

    int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
    int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    int const n_atoms1 = ctx.block_type_n_atoms[block_type1];
    int const n_atoms2 = ctx.block_type_n_atoms[block_type2];

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared =
        ([=](int, int, TMOL_SHARED_REF(Shared), TMOL_THREAD LKBallScoringData<Real>&) {});

    auto eval_interres_atom_pair_scores =
        ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_ROTAMER_CALCULATED_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared =
        ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    auto eval_intrares_atom_pair_scores =
        ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceDispatch,
        Dev,
        LKBallScoringData<Real>,
        LKBallScoringData<Real>,
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
        store_calculated_energies,
        lane);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol

// The kernel body above is the only user of these. lk_ball_pose_score.impl.hh
// keeps its own copies for the rotamer and derivative paths, which capture
// local state rather than a context struct; without this they would be
// redefinitions.
#undef LOAD_TILE_INVARIANT_INTERRES_DATA
#undef LOAD_INTERRES1_TILE_DATA_TO_SHARED
#undef LOAD_INTERRES2_TILE_DATA_TO_SHARED
#undef LOAD_INTERRES_DATA_FROM_SHARED
#undef EVAL_INTERRES_ATOM_PAIR_SCORES
#undef STORE_POSE_CALCULATED_ENERGIES
#undef STORE_ROTAMER_CALCULATED_ENERGIES
#undef LOAD_TILE_INVARIANT_INTRARES_DATA
#undef LOAD_INTRARES1_TILE_DATA_TO_SHARED
#undef LOAD_INTRARES2_TILE_DATA_TO_SHARED
#undef LOAD_INTRARES_DATA_FROM_SHARED
#undef EVAL_INTRARES_ATOM_PAIR_SCORES
