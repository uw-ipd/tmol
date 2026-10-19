#pragma once

// elec kernel bodies.
//
// Separated from elec_pose_score.impl.hh -- which holds the dispatch entry
// points and therefore reaches the torch tensor layer -- so that a Metal
// translation unit can include just the kernels. Everything named here is
// freestanding: the portable math layer, the view-generic helpers in elec.hh,
// and the device-templated workgroup machinery.
//
// Note the absence of device_operations.hh. The bodies reach the workgroup
// primitives through their DeviceDispatch template parameter, which is only
// resolved at instantiation, so the interface declaration -- which returns
// TPack and therefore needs torch -- is the caller's concern, not theirs.
//
// The bodies are templated on their context and shared-memory types, so the
// CPU and CUDA builds pass TView-backed state while a `.metal` entry passes
// MetalView-backed state and runs the same code.

#include <tmol/utility/portable/array.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/elec/potentials/elec.hh>
#include <tmol/score/elec/potentials/elec_ctx.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

#define MAX_N_CONN 12
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

using tmol::portable::Vec;

// Shared-memory block for one elec workgroup.
//
// Hoisted to namespace scope so a kernel body can take it by reference rather
// than declare it inline. Metal permits `threadgroup` declarations only at
// kernel-entry scope, so the entry declares the storage and passes it down;
// CUDA and CPU are unaffected, since the caller declares it either way (via
// SHARED_MEMORY, which is __shared__ under nvcc and empty on the host).
template <typename Real, int TILE, int MAX_CONN, typename launch_t>
union ElecSharedMem {
  CTA_REAL_REDUCE_T_TYPEDEF;

#if !TMOL_PORTABLE_METAL
  // A union whose members have non-trivial default constructors needs one of
  // its own. Metal has no such member here -- the reduction storage is empty
  // on that backend -- and a `threadgroup` variable may not be constructed at
  // all: any initializer turns into a guarded dynamic initialization, which
  // the Metal linker has no runtime for.
  ElecSharedMem() {}
#endif
  ElecBlockPairSharedData<Real, TILE, MAX_CONN> m;
  CTA_REAL_REDUCE_T_VARIABLE;
};

using tmol::portable::Vec;

#define SCORE_INTER_ELEC_ATOM_PAIR                                      \
  TMOL_DEVICE_FUNC(                                                     \
      int start_atom1,                                                  \
      int start_atom2,                                                  \
      int atom_tile_ind1,                                               \
      int atom_tile_ind2,                                               \
      TMOL_THREAD ElecScoringData<Real> const& inter_dat)                           \
      ->tmol::portable::array<Real, 1> {                                           \
    int separation = interres_count_pair_separation<TILE_SIZE>(         \
        inter_dat,                                                      \
        atom_tile_ind1,                                                 \
        atom_tile_ind2,                                                 \
        ctx.block_type_all_atoms_ligand_typed[inter_dat.r1.block_type]  \
            && ctx.block_type_all_atoms_ligand_typed[inter_dat.r2       \
                                                         .block_type]); \
    Real elec = elec_atom_energy_and_derivs(                            \
        atom_tile_ind1,                                                 \
        atom_tile_ind2,                                                 \
        start_atom1,                                                    \
        start_atom2,                                                    \
        inter_dat,                                                      \
        separation);                                                    \
    return {elec};                                                      \
  }

#define SCORE_INTRA_ELEC_ATOM_PAIR                                       \
  TMOL_DEVICE_FUNC(                                                      \
      int start_atom1,                                                   \
      int start_atom2,                                                   \
      int atom_tile_ind1,                                                \
      int atom_tile_ind2,                                                \
      TMOL_THREAD ElecScoringData<Real> const& intra_dat)                            \
      ->tmol::portable::array<Real, 1> {                                            \
    int const atom_ind1 = start_atom1 + atom_tile_ind1;                  \
    int const atom_ind2 = start_atom2 + atom_tile_ind2;                  \
    int const separation =                                               \
        ctx.block_type_intra_repr_path_distance[intra_dat.r1.block_type] \
                                               [atom_ind1][atom_ind2];   \
    Real elec = elec_atom_energy_and_derivs(                             \
        atom_tile_ind1,                                                  \
        atom_tile_ind2,                                                  \
        start_atom1,                                                     \
        start_atom2,                                                     \
        intra_dat,                                                       \
        separation);                                                     \
    return {elec};                                                       \
  }

#define LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED    \
  TMOL_DEVICE_FUNC(                                 \
      int pose_ind,                                 \
      TMOL_THREAD ElecSingleResData<Real>& r_dat,               \
      int n_atoms_to_load,                          \
      int start_atom) {                             \
    elec_load_block_coords_and_charges_into_shared< \
        DeviceDispatch,                             \
        D,                                          \
        nt,                                         \
        Real>(                                      \
        ctx.rot_coords,                             \
        ctx.block_type_partial_charge,              \
        pose_ind,                                   \
        r_dat,                                      \
        n_atoms_to_load,                            \
        start_atom, lane);                                \
  }

#define LOAD_BLOCK_INTO_SHARED                                                \
  TMOL_DEVICE_FUNC(                                                           \
      int pose_ind,                                                           \
      TMOL_THREAD ElecSingleResData<Real>& r_dat,                                         \
      int n_atoms_to_load,                                                    \
      int start_atom,                                                         \
      bool count_pair_striking_dist,                                          \
      TMOL_SHARED_PTR(unsigned char) conn_ats) {                                 \
    elec_load_block_into_shared<DeviceDispatch, D, nt, TILE_SIZE, Real, Int>( \
        ctx.rot_coords,                                                       \
        ctx.block_type_partial_charge,                                        \
        ctx.block_type_inter_repr_path_distance,                              \
        pose_ind,                                                             \
        r_dat,                                                                \
        n_atoms_to_load,                                                      \
        start_atom,                                                           \
        count_pair_striking_dist,                                             \
        conn_ats, lane);                                                            \
  }

#define LOAD_TILE_INVARIANT_INTERRES_DATA                                     \
  TMOL_DEVICE_FUNC(                                                           \
      int pose_ind,                                                           \
      int rot_ind1,                                                           \
      int rot_ind2,                                                           \
      int block_ind1,                                                         \
      int block_ind2,                                                         \
      int block_type1,                                                        \
      int block_type2,                                                        \
      int n_atoms1,                                                           \
      int n_atoms2,                                                           \
      TMOL_THREAD ElecScoringData<Real>& inter_dat,                                       \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                             \
    elec_load_tile_invariant_interres_data<DeviceDispatch, D, nt, Int, Real>( \
        ctx.rot_coord_offset,                                                 \
        ctx.pose_stack_near_blocks,                                           \
        ctx.block_type_n_interblock_bonds,                                    \
        ctx.block_type_atoms_forming_chemical_bonds,                          \
        ctx.pose_stack_inter_block_bondsep,                                   \
        ctx.global_params,                                                    \
        max_important_bond_separation,                                        \
        pose_ind,                                                             \
        rot_ind1,                                                             \
        rot_ind2,                                                             \
        block_ind1,                                                           \
        block_ind2,                                                           \
        block_type1,                                                          \
        block_type2,                                                          \
        n_atoms1,                                                             \
        n_atoms2,                                                             \
        inter_dat,                                                            \
        shared.m, lane);                                                            \
  }

#define LOAD_INTERRES1_TILE_DATA_TO_SHARED       \
  TMOL_DEVICE_FUNC(                              \
      int tile_ind,                              \
      int start_atom1,                           \
      int n_atoms_to_load1,                      \
      TMOL_THREAD ElecScoringData<Real>& inter_dat,          \
      TMOL_SHARED_REF(shared_mem_union) shared) {                \
    elec_load_interres1_tile_data_to_shared<     \
        DeviceDispatch,                          \
        D,                                       \
        nt,                                      \
        TILE_SIZE,                               \
        MAX_N_CONN,                              \
        Real,                                    \
        Int>(                                    \
        ctx.rot_coords,                          \
        ctx.block_type_partial_charge,           \
        ctx.block_type_inter_repr_path_distance, \
        tile_ind,                                \
        start_atom1,                             \
        n_atoms_to_load1,                        \
        inter_dat,                               \
        shared.m, lane);                               \
  }

#define LOAD_INTERRES2_TILE_DATA_TO_SHARED       \
  TMOL_DEVICE_FUNC(                              \
      int tile_ind,                              \
      int start_atom2,                           \
      int n_atoms_to_load2,                      \
      TMOL_THREAD ElecScoringData<Real>& inter_dat,          \
      TMOL_SHARED_REF(shared_mem_union) shared) {                \
    elec_load_interres2_tile_data_to_shared<     \
        DeviceDispatch,                          \
        D,                                       \
        nt,                                      \
        TILE_SIZE,                               \
        MAX_N_CONN,                              \
        Real,                                    \
        Int>(                                    \
        ctx.rot_coords,                          \
        ctx.block_type_partial_charge,           \
        ctx.block_type_inter_repr_path_distance, \
        tile_ind,                                \
        start_atom2,                             \
        n_atoms_to_load2,                        \
        inter_dat,                               \
        shared.m, lane);                               \
  }

#define LOAD_INTERRES_DATA_FROM_SHARED \
  TMOL_DEVICE_FUNC(int, int, TMOL_SHARED_REF(shared_mem_union), TMOL_THREAD ElecScoringData<Real>&) {}

#define EVAL_INTERRES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      TMOL_THREAD ElecScoringData<Real>& inter_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto elecE = tmol::score::common::InterResBlockEvaluation<            \
          ElecScoringData,                                                  \
          AllAtomPairSelector,                                              \
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
              score_inter_elec_atom_pair,                                   \
              inter_dat);                                                   \
      inter_dat.total_elec += tmol::portable::get<0>(elecE);                           \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                  \
        eval_scores_for_atom_pairs, lane);                                  \
  }

// In block-pair scoring mode, we are safe to simply write the calculated
// energies to the ctx.output tensor, since each block-pair energy is assigned
// to its own CTA: no need for an atomic-add call.
#define STORE_CALCULATED_POSE_ENERGIES                                       \
  TMOL_DEVICE_FUNC(                                                          \
      TMOL_THREAD ElecScoringData<Real>& score_dat, TMOL_SHARED_REF(shared_mem_union) shared) {          \
    auto reduce_energies = ([&](int tid) {                                   \
      Real const cta_total_elec =                                            \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(               \
              score_dat.total_elec,                                          \
              shared,                                                          \
              tmol::portable::plus_t<Real>(),                                  \
              lane);                                                           \
      if (tid == 0) {                                                        \
        if (!ctx.output_block_pair_energies) {                               \
          accumulate<D, Real>::add(                                          \
              ctx.output[0][score_dat.pose_ind][0][0], cta_total_elec);      \
        } else {                                                             \
          int const p = score_dat.pose_ind;                                  \
          int const b1 = score_dat.block_ind1;                               \
          int const b2 = score_dat.block_ind2;                               \
          ctx.output[0][p][b1][b2] = cta_total_elec;                         \
        }                                                                    \
      }                                                                      \
    });                                                                      \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                   \
        reduce_energies, lane);                                                \
  }

#define STORE_CALCULATED_ROTAMER_ENERGIES                                    \
  TMOL_DEVICE_FUNC(                                                          \
      TMOL_THREAD ElecScoringData<Real>& score_dat, TMOL_SHARED_REF(shared_mem_union) shared) {          \
    auto reduce_energies = ([&](int tid) {                                   \
      Real const cta_total_elec =                                            \
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(               \
              score_dat.total_elec,                                          \
              shared,                                                          \
              tmol::portable::plus_t<Real>(),                                  \
              lane);                                                           \
      if (tid == 0) {                                                        \
        ctx.output[0][cta] = cta_total_elec;                                 \
      }                                                                      \
    });                                                                      \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                   \
        reduce_energies, lane);                                                \
  }

#define LOAD_TILE_INVARIANT_INTRARES_DATA                                     \
  TMOL_DEVICE_FUNC(                                                           \
      int pose_ind,                                                           \
      int rot_ind1,                                                           \
      int block_ind1,                                                         \
      int block_type1,                                                        \
      int n_atoms1,                                                           \
      TMOL_THREAD ElecScoringData<Real>& intra_dat,                                       \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                             \
    elec_load_tile_invariant_intrares_data<DeviceDispatch, D, nt, Int, Real>( \
        ctx.rot_coord_offset,                                                 \
        ctx.global_params,                                                    \
        max_important_bond_separation,                                        \
        pose_ind,                                                             \
        rot_ind1,                                                             \
        block_ind1,                                                           \
        block_type1,                                                          \
        n_atoms1,                                                             \
        intra_dat,                                                            \
        shared.m, lane);                                                            \
  }

#define LOAD_INTRARES1_TILE_DATA_TO_SHARED   \
  TMOL_DEVICE_FUNC(                          \
      int tile_ind,                          \
      int start_atom1,                       \
      int n_atoms_to_load1,                  \
      TMOL_THREAD ElecScoringData<Real>& intra_dat,      \
      TMOL_SHARED_REF(shared_mem_union) shared) {            \
    elec_load_intrares1_tile_data_to_shared< \
        DeviceDispatch,                      \
        D,                                   \
        nt,                                  \
        TILE_SIZE,                           \
        MAX_N_CONN,                          \
        Real>(                               \
        ctx.rot_coords,                      \
        ctx.block_type_partial_charge,       \
        tile_ind,                            \
        start_atom1,                         \
        n_atoms_to_load1,                    \
        intra_dat,                           \
        shared.m, lane);                           \
  }

#define LOAD_INTRARES2_TILE_DATA_TO_SHARED   \
  TMOL_DEVICE_FUNC(                          \
      int tile_ind,                          \
      int start_atom2,                       \
      int n_atoms_to_load2,                  \
      TMOL_THREAD ElecScoringData<Real>& intra_dat,      \
      TMOL_SHARED_REF(shared_mem_union) shared) {            \
    elec_load_intrares2_tile_data_to_shared< \
        DeviceDispatch,                      \
        D,                                   \
        nt,                                  \
        TILE_SIZE,                           \
        MAX_N_CONN,                          \
        Real>(                               \
        ctx.rot_coords,                      \
        ctx.block_type_partial_charge,       \
        tile_ind,                            \
        start_atom2,                         \
        n_atoms_to_load2,                    \
        intra_dat,                           \
        shared.m, lane);                           \
  }

#define LOAD_INTRARES_DATA_FROM_SHARED              \
  TMOL_DEVICE_FUNC(                                 \
      int tile_ind1,                                \
      int tile_ind2,                                \
      TMOL_SHARED_REF(shared_mem_union) shared,     \
      TMOL_THREAD ElecScoringData<Real>& intra_dat) {           \
    elec_load_intrares_data_from_shared(            \
        tile_ind1, tile_ind2, shared.m, intra_dat); \
  }

#define EVAL_INTRARES_ATOM_PAIR_SCORES                                      \
  TMOL_DEVICE_FUNC(                                                         \
      TMOL_THREAD ElecScoringData<Real>& intra_dat, int start_atom1, int start_atom2) { \
    auto eval_scores_for_atom_pairs = ([&](int tid) {                       \
      auto elecE = tmol::score::common::IntraResBlockEvaluation<            \
          ElecScoringData,                                                  \
          AllAtomPairSelector,                                              \
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
              score_intra_elec_atom_pair,                                   \
              intra_dat);                                                   \
      intra_dat.total_elec += tmol::portable::get<0>(elecE);                           \
    });                                                                     \
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(                  \
        eval_scores_for_atom_pairs, lane);                                  \
  }

// Kernel body, extracted from a device lambda so that a backend whose entry
// points cannot be closures -- Metal's `kernel void` functions -- can call the
// same code CPU and CUDA run. State arrives through ElecCtx instead of an
// implicit [=] capture, and shared memory is a parameter because Metal permits
// `threadgroup` declarations only at kernel scope.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void elec_eval_energies_by_block_2(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t> shared_mem_union;
  auto elec_atom_energy_and_derivs =
      ([=] TMOL_DEVICE_FUNC(
           int atom_tile_ind1,
           int atom_tile_ind2,
           int,
           int,
           TMOL_THREAD ElecScoringData<Real> const& score_dat,
           int cp_separation) {
        return elec_atom_energy(
            atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
      });

  auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

  auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

  auto load_block_coords_and_params_into_shared =
      ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

  auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

  int const max_important_bond_separation = 4;

  int const pose_ind = cta / (ctx.max_n_upper_triangle_inds);
  int const block_ind_pair = cta % (ctx.max_n_upper_triangle_inds);

  auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
      block_ind_pair, ctx.max_n_blocks + 1);
  int const block_ind1 = common::get<0>(upper_triangle_ind);
  int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

  // We do not have to kill half of our thread blocks simply because they
  // represent the lower triangle now that we're using upper-triangle indices,
  // So no need for this check:
  // if (block_ind1 > block_ind2) {
  //   return;
  // }

  // But we still kill everything that involves non-neighboring blocks
  // and that can be a lot!
  if (!ctx.use_shared_compact_block_neighbors
      && ctx.scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
    return;
  }

  int const rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
  int const rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];

  if (rot_ind1 < 0 || rot_ind2 < 0) {
    return;
  }

  int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
  int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];

  if (block_type1 < 0 || block_type2 < 0) {
    // I think this cannot happen
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
      ([=] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

  auto store_calculated_energies = ([=] STORE_CALCULATED_POSE_ENERGIES);

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
      D,
      ElecScoringData<Real>,
      ElecScoringData<Real>,
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

// Kernel body, extracted from a device lambda so that a backend whose entry
// points cannot be closures -- Metal's `kernel void` functions -- can call the
// same code CPU and CUDA run. State arrives through ElecCtx instead of an
// implicit [=] capture, and shared memory is a parameter because Metal permits
// `threadgroup` declarations only at kernel scope.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void elec_eval_energies(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t> shared_mem_union;
  auto elec_atom_energy_and_derivs =
      ([=] TMOL_DEVICE_FUNC(
           int atom_tile_ind1,
           int atom_tile_ind2,
           int start_atom1,
           int start_atom2,
           TMOL_THREAD ElecScoringData<Real> const& score_dat,
           int cp_separation) {
        return elec_atom_energy_and_derivs_full<Real, D>(
            atom_tile_ind1,
            atom_tile_ind2,
            start_atom1,
            start_atom2,
            score_dat,
            cp_separation,
            ctx.dV_dcoords);
      });

  auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

  auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

  auto load_block_coords_and_params_into_shared =
      ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

  auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

  int const max_important_bond_separation = 4;

  int const pose_ind = cta / (ctx.max_n_upper_triangle_inds);
  int const block_ind_pair = cta % (ctx.max_n_upper_triangle_inds);

  auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
      block_ind_pair, ctx.max_n_blocks + 1);
  int const block_ind1 = common::get<0>(upper_triangle_ind);
  int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

  // We do not have to kill half of our thread blocks simply because they
  // represent the lower triangle now that we're using upper-triangle indices
  // if (block_ind1 > block_ind2) {
  //   return;
  // }

  // But we still kill everything that involves non-neighboring blocks
  // and that can be a lot!
  if (!ctx.use_shared_compact_block_neighbors
      && ctx.scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
    return;
  }

  int const rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
  int const rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];

  if (rot_ind1 < 0 || rot_ind2 < 0) {
    return;
  }

  int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
  int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];

  if (block_type1 < 0 || block_type2 < 0) {
    // I think this cannot happen
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
      ([=] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

  auto store_calculated_energies = ([=] STORE_CALCULATED_POSE_ENERGIES);

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
      D,
      ElecScoringData<Real>,
      ElecScoringData<Real>,
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

// Kernel body, extracted from a device lambda so that a backend whose entry
// points cannot be closures -- Metal's `kernel void` functions -- can call the
// same code CPU and CUDA run. State arrives through ElecCtx instead of an
// implicit [=] capture, and shared memory is a parameter because Metal permits
// `threadgroup` declarations only at kernel scope.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void elec_eval_derivs_2(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t> shared_mem_union;
  auto elec_atom_energy_and_derivs =
      ([=] TMOL_DEVICE_FUNC(
           int atom_tile_ind1,
           int atom_tile_ind2,
           int start_atom1,
           int start_atom2,
           TMOL_THREAD ElecScoringData<Real> const& score_dat,
           int cp_separation) {
        elec_atom_derivs<Real, D>(
            atom_tile_ind1,
            atom_tile_ind2,
            start_atom1,
            start_atom2,
            score_dat,
            cp_separation,
            ctx.dTdV[0][score_dat.pose_ind][score_dat.block_ind1]
                    [score_dat.block_ind2],
            ctx.dV_dcoords);
        return 0.0;
      });
  auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

  auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

  auto load_block_coords_and_params_into_shared =
      ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

  auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

  int const max_important_bond_separation = 4;
  int const pose_ind = cta / (ctx.max_n_upper_triangle_inds);
  int const block_ind_pair = cta % (ctx.max_n_upper_triangle_inds);

  auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
      block_ind_pair, ctx.max_n_blocks + 1);
  int const block_ind1 = common::get<0>(upper_triangle_ind);
  int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

  int const rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
  int const rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];

  if (ctx.scratch_rot_neighbors[pose_ind][block_ind1][block_ind2] == 0) {
    return;
  }

  if (ctx.dTdV[0][pose_ind][block_ind1][block_ind2] == 0) {
    return;
  }

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
      ([=] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

  auto store_calculated_energies =
      ([=](TMOL_THREAD ElecScoringData<Real>& score_dat, TMOL_SHARED_REF(shared_mem_union) shared) {
        ;  // no op when what we're computing are the gradients ()
      });

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
      D,
      ElecScoringData<Real>,
      ElecScoringData<Real>,
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

// Kernel body, extracted from a device lambda so that a backend whose entry
// points cannot be closures -- Metal's `kernel void` functions -- can call the
// same code CPU and CUDA run. State arrives through ElecCtx instead of an
// implicit [=] capture, and shared memory is a parameter because Metal permits
// `threadgroup` declarations only at kernel scope.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void elec_eval_energies_by_block(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t> shared_mem_union;
  auto elec_atom_energy_and_derivs =
      ([=] TMOL_DEVICE_FUNC(
           int atom_tile_ind1,
           int atom_tile_ind2,
           int,
           int,
           TMOL_THREAD ElecScoringData<Real> const& score_dat,
           int cp_separation) {
        return elec_atom_energy(
            atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
      });

  auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

  auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

  auto load_block_coords_and_params_into_shared =
      ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

  auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

  int const max_important_bond_separation = 4;

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
      ([=] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

  auto store_calculated_energies =
      ([=] STORE_CALCULATED_ROTAMER_ENERGIES);

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
      D,
      ElecScoringData<Real>,
      ElecScoringData<Real>,
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

// Kernel body, extracted from a device lambda so that a backend whose entry
// points cannot be closures -- Metal's `kernel void` functions -- can call the
// same code CPU and CUDA run. State arrives through ElecCtx instead of an
// implicit [=] capture, and shared memory is a parameter because Metal permits
// `threadgroup` declarations only at kernel scope.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void elec_eval_derivs(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<D> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef ElecSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t> shared_mem_union;
  auto elec_atom_energy_and_derivs =
      ([=] TMOL_DEVICE_FUNC(
           int atom_tile_ind1,
           int atom_tile_ind2,
           int start_atom1,
           int start_atom2,
           TMOL_THREAD ElecScoringData<Real> const& score_dat,
           int cp_separation) {
        elec_atom_derivs<Real, D>(
            atom_tile_ind1,
            atom_tile_ind2,
            start_atom1,
            start_atom2,
            score_dat,
            cp_separation,
            ctx.dTdV[0][cta],
            ctx.dV_dcoords);
        return 0.0;
      });

  // TEST!
  auto score_inter_elec_atom_pair = ([=] SCORE_INTER_ELEC_ATOM_PAIR);

  auto score_intra_elec_atom_pair = ([=] SCORE_INTRA_ELEC_ATOM_PAIR);

  auto load_block_coords_and_params_into_shared =
      ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

  auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

  int const max_important_bond_separation = 4;
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
      ([=] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

  auto store_calculated_energies =
      ([=](TMOL_THREAD ElecScoringData<Real>& score_dat, TMOL_SHARED_REF(shared_mem_union) shared) {
        ;  // no op for gradients ()
      });

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
      D,
      ElecScoringData<Real>,
      ElecScoringData<Real>,
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
}  // namespace elec
}  // namespace score
}  // namespace tmol
