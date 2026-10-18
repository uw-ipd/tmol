#pragma once

// hbond kernel bodies.
//
// Separated from hbond_pose_score.impl.hh -- which holds the dispatch entry
// points and therefore reaches the torch tensor layer -- so that a Metal
// translation unit can include just the kernels. Everything named here is
// freestanding: the portable math layer, the view-generic helpers in hbond.hh,
// and the device-templated workgroup machinery.
//
// Note the absence of device_operations.hh. The bodies reach the workgroup
// primitives through their DeviceDispatch template parameter, which is only
// resolved at instantiation, so the interface declaration -- which returns
// TPack and therefore needs torch -- is the caller's concern, not theirs.
//
// The bodies are templated on their context and shared-memory types, so the
// CPU and CUDA builds pass TView-backed state while a `.metal` entry passes
// MetalView-backed state and runs the same code. The whole-pose dispatch
// bodies still live in hbond_pose_score.impl.hh, but they expand the same
// macros against the same context, so the tile machinery below is written
// once for both families.

#include <tmol/utility/portable/array.hh>
#include <tmol/utility/portable/reduce_ops.hh>
#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/hbond/potentials/hbond.hh>
#include <tmol/score/hbond/potentials/hbond_ctx.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 12
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using tmol::score::common::accumulate;
using tmol::portable::Vec;

// A compile-time stand-in for std::true_type / std::false_type.
//
// The whole-pose body takes whether to accumulate derivatives as a tag rather
// than a flag so the caller can make it compile-time; CUDA passes the standard
// tags, but MSL has no <type_traits>, so a Metal entry passes this. Converting
// it is constexpr, which is what lets the optimizer drop the unused half of
// HBOND_ATOM_ENERGY -- worth 17% of the entry's occupancy, and considerably
// more than that in wall time.
template <bool B>
struct deriv_tag {
  TMOL_HD constexpr operator bool() const { return B; }
};

// Shared-memory block for one hbond workgroup.
//
// Hoisted to namespace scope so a kernel body can take it by reference rather
// than declare it inline. Metal permits `threadgroup` declarations only at
// kernel-entry scope, so the entry declares the storage and passes it down;
// CUDA and CPU are unaffected, since the caller declares it either way (via
// SHARED_MEMORY, which is __shared__ under nvcc and empty on the host).
template <typename Real, int TILE, int MAX_CONN, typename launch_t>
union HBondSharedMem {
  CTA_REAL_REDUCE_T_TYPEDEF;

#if !TMOL_PORTABLE_METAL
  // A union whose members have non-trivial default constructors needs one of
  // its own. Metal has no such member here -- the reduction storage is empty
  // on that backend -- and a `threadgroup` variable may not be constructed at
  // all: any initializer turns into a guarded dynamic initialization, which
  // the Metal linker has no runtime for.
  HBondSharedMem() {}
#endif
  HBondBlockPairSharedData<Real, TILE, MAX_CONN> m;
  CTA_REAL_REDUCE_T_VARIABLE;
};

// done
#define HBOND_ATOM_ENERGY                                      \
  TMOL_DEVICE_FUNC(                                            \
      int don_ind,                                             \
      int acc_ind,                                             \
      int don_tile_ind,                                        \
      int acc_tile_ind,                                        \
      int don_start,                                           \
      int acc_start,                                           \
      TMOL_THREAD HBondSingleResData<Real> const& don_dat,                 \
      TMOL_THREAD HBondSingleResData<Real> const& acc_dat,                 \
      TMOL_THREAD HBondRotPairData<Real> const& respair_dat,     \
      int cp_separation) {                                     \
    if (cp_separation < 5) {                                   \
      return Real(0.0);                                        \
    }                                                          \
    if (accumulate_derivs) {                                   \
      Real val =                                               \
          hbond_atom_energy_and_derivs_full<TILE_SIZE, Dev, Real, Int>( \
              don_ind,                                         \
              acc_ind,                                         \
              don_tile_ind,                                    \
              acc_tile_ind,                                    \
              don_start,                                       \
              acc_start,                                       \
              don_dat,                                         \
              acc_dat,                                         \
              respair_dat,                                     \
              cp_separation,                                   \
              ctx.derived_coords,                                  \
              ctx.derived_atom_inds,                               \
              ctx.pair_params,                                     \
              ctx.pair_polynomials,                                \
              ctx.dV_dcoords);                                     \
      return val;                                              \
    } else {                                                   \
      Real val = hbond_atom_energy_full<TILE_SIZE, Real, Int>( \
          don_ind,                                             \
          acc_ind,                                             \
          don_tile_ind,                                        \
          acc_tile_ind,                                        \
          don_start,                                           \
          acc_start,                                           \
          don_dat,                                             \
          acc_dat,                                             \
          respair_dat,                                         \
          cp_separation,                                       \
          ctx.derived_coords,                                      \
          ctx.derived_atom_inds,                                   \
          ctx.pair_params,                                         \
          ctx.pair_polynomials);                                   \
      return val;                                              \
    }                                                          \
  }

// done
#define SCORE_INTER_HBOND_ATOM_PAIR                                          \
  TMOL_DEVICE_FUNC(                                                          \
      int don_start,                                                         \
      int acc_start,                                                         \
      int don_ind,                                                           \
      int acc_ind,                                                           \
      TMOL_THREAD HBondScoringData<Real> const& inter_dat,                     \
      bool donor_first)                                                      \
      ->tmol::portable::array<Real, 1> {                                                \
    int don_tile_ind =                                                       \
        (donor_first ? inter_dat.r1 : inter_dat.r2).donH_tile_inds[don_ind]; \
    int acc_tile_ind =                                                       \
        (donor_first ? inter_dat.r2 : inter_dat.r1).acc_tile_inds[acc_ind];  \
    TMOL_THREAD HBondSingleResData<Real> const& don_dat =                \
        donor_first ? inter_dat.r1 : inter_dat.r2;                             \
    TMOL_THREAD HBondSingleResData<Real> const& acc_dat =                \
        donor_first ? inter_dat.r2 : inter_dat.r1;                             \
    auto const Hxyz = coord_from_shared(don_dat.coords, don_tile_ind);       \
    auto const Axyz = coord_from_shared(acc_dat.coords, acc_tile_ind);       \
    Real const cutoff = inter_dat.pair_data.global_params.max_ha_dis;        \
    if (!((Hxyz - Axyz).squaredNorm() < cutoff * cutoff)) {                  \
      return {Real(0)};                                                      \
    }                                                                        \
    int separation = interres_count_pair_separation<TILE_SIZE>(              \
        inter_dat,                                                           \
        (donor_first ? don_ind : acc_ind),                                   \
        (donor_first ? acc_ind : don_ind));                                  \
    Real hbond = hbond_atom_energy(                                          \
        don_ind,                                                             \
        acc_ind,                                                             \
        don_tile_ind,                                                        \
        acc_tile_ind,                                                        \
        don_start,                                                           \
        acc_start,                                                           \
        donor_first ? inter_dat.r1 : inter_dat.r2,                           \
        donor_first ? inter_dat.r2 : inter_dat.r1,                           \
        inter_dat.pair_data,                                                 \
        separation);                                                         \
    return {hbond};                                                          \
  }

// done
#define SCORE_INTRA_HBOND_ATOM_PAIR                                          \
  TMOL_DEVICE_FUNC(                                                          \
      int don_start,                                                         \
      int acc_start,                                                         \
      int don_ind,                                                           \
      int acc_ind,                                                           \
      TMOL_THREAD HBondScoringData<Real> const& intra_dat,                     \
      bool donor_first)                                                      \
      ->tmol::portable::array<Real, 1> {                                                \
    int don_tile_ind =                                                       \
        (donor_first ? intra_dat.r1 : intra_dat.r2).donH_tile_inds[don_ind]; \
    int acc_tile_ind =                                                       \
        (donor_first ? intra_dat.r2 : intra_dat.r1).acc_tile_inds[acc_ind];  \
    int const don_atom_ind = don_start + don_tile_ind;                       \
    int const acc_atom_ind = acc_start + acc_tile_ind;                       \
    int const separation =                                                   \
        ctx.block_type_path_distance[intra_dat.r1.block_type][don_atom_ind]      \
                                [acc_atom_ind];                              \
    Real hbond = hbond_atom_energy(                                          \
        don_ind,                                                             \
        acc_ind,                                                             \
        don_tile_ind,                                                        \
        acc_tile_ind,                                                        \
        don_start,                                                           \
        acc_start,                                                           \
        donor_first ? intra_dat.r1 : intra_dat.r2,                           \
        donor_first ? intra_dat.r2 : intra_dat.r1,                           \
        intra_dat.pair_data,                                                 \
        separation);                                                         \
    return {hbond};                                                          \
  }

// done
#define LOAD_TILE_INVARIANT_INTERRES_DATA                             \
  TMOL_DEVICE_FUNC(                                                   \
      int pose_ind,                                                   \
      int rot_ind1,                                                   \
      int rot_ind2,                                                   \
      int block_ind1,                                                 \
      int block_ind2,                                                 \
      int block_type1,                                                \
      int block_type2,                                                \
      int n_atoms1,                                                   \
      int n_atoms2,                                                   \
      TMOL_THREAD HBondScoringData<Real>& inter_dat,                    \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                     \
    hbond_load_tile_invariant_interres_data<DeviceDispatch, Dev, nt>( \
        ctx.rot_coord_offset,                                             \
        ctx.pose_stack_near_blocks,                                       \
        ctx.pose_stack_inter_block_bondsep,                               \
        ctx.block_type_n_interblock_bonds,                                \
        ctx.block_type_atoms_forming_chemical_bonds,                      \
        ctx.global_params,                                                \
        max_important_bond_separation,                                \
        pose_ind,                                                     \
        rot_ind1,                                                     \
        rot_ind2,                                                     \
        block_ind1,                                                   \
        block_ind2,                                                   \
        block_type1,                                                  \
        block_type2,                                                  \
        n_atoms1,                                                     \
        n_atoms2,                                                     \
        inter_dat,                                                    \
        shared.m,                                                     \
        lane);                                                        \
  }

// done
#define LOAD_INTERRES1_TILE_DATA_TO_SHARED                             \
  TMOL_DEVICE_FUNC(                                                    \
      int tile_ind,                                                    \
      int start_atom1,                                                 \
      int n_atoms_to_load1,                                            \
      TMOL_THREAD HBondScoringData<Real>& inter_dat,                     \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                      \
    hbond_load_interres1_tile_data_to_shared<DeviceDispatch, Dev, nt>( \
        ctx.rot_coords,                                                    \
        ctx.block_type_tile_n_donH,                                        \
        ctx.block_type_tile_n_acc,                                         \
        ctx.block_type_tile_donH_inds,                                     \
        ctx.block_type_tile_acc_inds,                                      \
        ctx.block_type_tile_donor_type,                                    \
        ctx.block_type_tile_acceptor_type,                                 \
        ctx.block_type_tile_hybridization,                                 \
        ctx.block_type_path_distance,                                      \
        tile_ind,                                                      \
        start_atom1,                                                   \
        n_atoms_to_load1,                                              \
        inter_dat,                                                     \
        shared.m,                                                      \
        lane);                                                         \
  }

// done
#define LOAD_INTERRES2_TILE_DATA_TO_SHARED                             \
  TMOL_DEVICE_FUNC(                                                    \
      int tile_ind,                                                    \
      int start_atom2,                                                 \
      int n_atoms_to_load2,                                            \
      TMOL_THREAD HBondScoringData<Real>& inter_dat,                     \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                      \
    hbond_load_interres2_tile_data_to_shared<DeviceDispatch, Dev, nt>( \
        ctx.rot_coords,                                                    \
        ctx.block_type_tile_n_donH,                                        \
        ctx.block_type_tile_n_acc,                                         \
        ctx.block_type_tile_donH_inds,                                     \
        ctx.block_type_tile_acc_inds,                                      \
        ctx.block_type_tile_donor_type,                                    \
        ctx.block_type_tile_acceptor_type,                                 \
        ctx.block_type_tile_hybridization,                                 \
        ctx.block_type_path_distance,                                      \
        tile_ind,                                                      \
        start_atom2,                                                   \
        n_atoms_to_load2,                                              \
        inter_dat,                                                     \
        shared.m,                                                      \
        lane);                                                         \
  }

// done
#define LOAD_INTERRES_DATA_FROM_SHARED              \
  TMOL_DEVICE_FUNC(                                 \
      int,                                          \
      int,                                          \
      TMOL_SHARED_REF(shared_mem_union),            \
      TMOL_THREAD HBondScoringData<Real>&) {}

// done
#define EVAL_INTERRES_ATOM_PAIR_SCORES                                     \
  TMOL_DEVICE_FUNC(                                                        \
      TMOL_THREAD HBondScoringData<Real>& inter_dat,                         \
      int start_atom1,                                                     \
      int start_atom2) {                                                   \
    eval_interres_don_acc_pair_energies<DeviceDispatch, Dev, nt>(          \
        inter_dat,                                                         \
        start_atom1,                                                       \
        start_atom2,                                                       \
        score_inter_hbond_atom_pair,                                       \
        lane);                                                             \
  }

// done
#define STORE_POSE_CALCULATED_ENERGIES                                         \
  TMOL_DEVICE_FUNC(                                                            \
      TMOL_THREAD HBondScoringData<Real>& score_dat,                           \
      TMOL_SHARED_REF(shared_mem_union) shared) {                              \
    auto reduce_energies = ([&](int tid) {                                     \
      Real const cta_total_hbond =                                             \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(               \
              score_dat.pair_data.total_hbond,                                 \
              shared,                                                          \
              tmol::portable::plus_t<Real>(),                                   \
              lane);                                                            \
      if (tid == 0) {                                                          \
        if (ctx.output_block_pair_energies) {                                      \
          int const p = score_dat.pair_data.pose_ind;                          \
          int const a_ind = score_dat.r1.block_ind;                            \
          int const b_ind = score_dat.r2.block_ind;                            \
          int const b1 = min(a_ind, b_ind);                                    \
          int const b2 = max(a_ind, b_ind);                                    \
          ctx.output[0][p][b1][b2] = cta_total_hbond;                              \
        } else {                                                               \
          int const p = score_dat.pair_data.pose_ind;                          \
          accumulate<Dev, Real>::add(ctx.output[0][p][0][0], cta_total_hbond);     \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(                   \
        reduce_energies, lane);                                                \
  }

#define STORE_ROTAMER_CALCULATED_ENERGIES                                      \
  TMOL_DEVICE_FUNC(                                                            \
      TMOL_THREAD HBondScoringData<Real>& score_dat,                           \
      TMOL_SHARED_REF(shared_mem_union) shared) {                              \
    auto reduce_energies = ([&](int tid) {                                     \
      Real const cta_total_hbond =                                             \
          DeviceDispatch<Dev>::template reduce_in_workgroup<nt>(               \
              score_dat.pair_data.total_hbond,                                 \
              shared,                                                          \
              tmol::portable::plus_t<Real>(),                                   \
              lane);                                                            \
      if (tid == 0) {                                                          \
        ctx.output[0][cta] = cta_total_hbond;                                      \
      }                                                                        \
    });                                                                        \
    DeviceDispatch<Dev>::template for_each_in_workgroup<nt>(                   \
        reduce_energies, lane);                                                \
  }

// done
#define LOAD_TILE_INVARIANT_INTRARES_DATA                             \
  TMOL_DEVICE_FUNC(                                                   \
      int pose_ind,                                                   \
      int rot_ind1,                                                   \
      int block_ind1,                                                 \
      int block_type1,                                                \
      int n_atoms1,                                                   \
      TMOL_THREAD HBondScoringData<Real>& intra_dat,                    \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                     \
    hbond_load_tile_invariant_intrares_data<DeviceDispatch, Dev, nt>( \
        ctx.rot_coord_offset,                                             \
        ctx.global_params,                                                \
        max_important_bond_separation,                                \
        pose_ind,                                                     \
        rot_ind1,                                                     \
        block_ind1,                                                   \
        block_type1,                                                  \
        n_atoms1,                                                     \
        intra_dat,                                                    \
        shared.m,                                                     \
        lane);                                                        \
  }

// done
#define LOAD_INTRARES1_TILE_DATA_TO_SHARED                             \
  TMOL_DEVICE_FUNC(                                                    \
      int tile_ind,                                                    \
      int start_atom1,                                                 \
      int n_atoms_to_load1,                                            \
      TMOL_THREAD HBondScoringData<Real>& intra_dat,                     \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                      \
    hbond_load_intrares1_tile_data_to_shared<DeviceDispatch, Dev, nt>( \
        ctx.rot_coords,                                                    \
        ctx.block_type_tile_n_donH,                                        \
        ctx.block_type_tile_n_acc,                                         \
        ctx.block_type_tile_donH_inds,                                     \
        ctx.block_type_tile_acc_inds,                                      \
        ctx.block_type_tile_donor_type,                                    \
        ctx.block_type_tile_acceptor_type,                                 \
        ctx.block_type_tile_hybridization,                                 \
        tile_ind,                                                      \
        start_atom1,                                                   \
        n_atoms_to_load1,                                              \
        intra_dat,                                                     \
        shared.m,                                                      \
        lane);                                                         \
  }

// done
#define LOAD_INTRARES2_TILE_DATA_TO_SHARED                             \
  TMOL_DEVICE_FUNC(                                                    \
      int tile_ind,                                                    \
      int start_atom2,                                                 \
      int n_atoms_to_load2,                                            \
      TMOL_THREAD HBondScoringData<Real>& intra_dat,                     \
      TMOL_SHARED_REF(shared_mem_union) shared) {                                      \
    hbond_load_intrares2_tile_data_to_shared<DeviceDispatch, Dev, nt>( \
        ctx.rot_coords,                                                    \
        ctx.block_type_tile_n_donH,                                        \
        ctx.block_type_tile_n_acc,                                         \
        ctx.block_type_tile_donH_inds,                                     \
        ctx.block_type_tile_acc_inds,                                      \
        ctx.block_type_tile_donor_type,                                    \
        ctx.block_type_tile_acceptor_type,                                 \
        ctx.block_type_tile_hybridization,                                 \
        tile_ind,                                                      \
        start_atom2,                                                   \
        n_atoms_to_load2,                                              \
        intra_dat,                                                     \
        shared.m,                                                      \
        lane);                                                         \
  }

// done
#define LOAD_INTRARES_DATA_FROM_SHARED               \
  TMOL_DEVICE_FUNC(                                  \
      int tile_ind1,                                 \
      int tile_ind2,                                 \
      TMOL_SHARED_REF(shared_mem_union) shared,                      \
      TMOL_THREAD HBondScoringData<Real>& intra_dat) { \
    hbond_load_intrares_data_from_shared(            \
        tile_ind1, tile_ind2, shared.m, intra_dat);  \
  }

// done
#define EVAL_INTRARES_ATOM_PAIR_SCORES                                     \
  TMOL_DEVICE_FUNC(                                                        \
      TMOL_THREAD HBondScoringData<Real>& intra_dat,                         \
      int start_atom1,                                                     \
      int start_atom2) {                                                   \
    eval_intrares_don_acc_pair_energies<DeviceDispatch, Dev, nt>(          \
        intra_dat,                                                         \
        start_atom1,                                                       \
        start_atom2,                                                       \
        score_intra_hbond_atom_pair,                                       \
        lane);                                                             \
  }

// Whole-pose scoring, extracted from a device lambda so that a backend whose
// entry points cannot be closures -- Metal's `kernel void` functions -- can
// call the identical body.
//
// One workgroup per (pose, block-pair) position in the upper triangle. On
// CUDA the caller may instantiate PairMode to split inter- and intra-residue
// pairs into two launches; DerivTag is a std::true_type / std::false_type when
// the caller knows at compile time whether derivatives are wanted and a plain
// bool otherwise, which is exactly what the lambda it replaced did.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    tmol::score::common::TilePairMode PairMode,
    typename Ctx,
    typename Shared,
    typename DerivTag>
TMOL_HD void hbond_pose_eval_energies(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    DerivTag derivative_tag,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
      shared_mem_union;
  bool const accumulate_derivs = static_cast<bool>(derivative_tag);

  auto hbond_atom_energy = ([ =, &ctx ] HBOND_ATOM_ENERGY);

  auto score_inter_hbond_atom_pair = ([ =, &ctx ] SCORE_INTER_HBOND_ATOM_PAIR);

  auto score_intra_hbond_atom_pair = ([ =, &ctx ] SCORE_INTRA_HBOND_ATOM_PAIR);

  int const max_important_bond_separation = 4;

  int const pose_ind = cta / ctx.max_n_upper_triangle_inds;
  int const block_ind_pair = cta % ctx.max_n_upper_triangle_inds;

  // We do not have to kill half of our thread blocks simply because they
  // represent the lower triangle now that we're using upper-triangle indices
  auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
      block_ind_pair, ctx.max_n_blocks + 1);

  int const block_ind1 = common::get<0>(upper_triangle_ind);
  int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

  if constexpr (PairMode == tmol::score::common::TilePairMode::Inter) {
    if (block_ind1 == block_ind2) return;
  } else if constexpr (PairMode == tmol::score::common::TilePairMode::Intra) {
    if (block_ind1 != block_ind2) return;
  }

  // We still kill CTAs targetting non-neighboring block pairs, though, and
  // that can be a lot. dispatch_indices is the block-neighbor table for this
  // kernel family.
  if (!ctx.use_shared_compact_block_neighbors
      && ctx.dispatch_indices[pose_ind][block_ind1][block_ind2] == 0) {
    return;
  }
  int const rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
  int const rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];

  int const block_type1 = ctx.block_type_ind_for_rot[rot_ind1];
  int const block_type2 = ctx.block_type_ind_for_rot[rot_ind2];

  if (block_type1 < 0 || block_type2 < 0) {
    return;
  }

  int const n_atoms1 = ctx.block_type_n_atoms[block_type1];
  int const n_atoms2 = ctx.block_type_n_atoms[block_type2];

  auto load_tile_invariant_interres_data =
      ([ =, &ctx ] LOAD_TILE_INVARIANT_INTERRES_DATA);

  auto load_interres1_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

  auto load_interres2_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

  auto load_interres_data_from_shared =
      ([ =, &ctx ] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([ =, &ctx ] EVAL_INTERRES_ATOM_PAIR_SCORES);

  auto store_calculated_energies = ([ =, &ctx ] STORE_POSE_CALCULATED_ENERGIES);

  auto load_tile_invariant_intrares_data =
      ([ =, &ctx ] LOAD_TILE_INVARIANT_INTRARES_DATA);

  auto load_intrares1_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

  auto load_intrares2_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

  auto load_intrares_data_from_shared =
      ([ =, &ctx ] LOAD_INTRARES_DATA_FROM_SHARED);

  auto eval_intrares_atom_pair_scores =
      ([ =, &ctx ] EVAL_INTRARES_ATOM_PAIR_SCORES);

  tmol::score::common::tile_evaluate_rot_pair<
      DeviceDispatch,
      Dev,
      HBondScoringData<Real>,
      HBondScoringData<Real>,
      Real,
      TILE_SIZE,
      PairMode>(
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

// Whole-pose derivatives, extracted from a device lambda for the same reason
// as the forward pass above.
//
// One workgroup per (pose, block-pair). Nothing is stored: the atom-energy
// hook accumulates straight into ctx.dV_dcoords, weighted by the incoming
// ctx.dTdV cell, and returns zero so the tile driver's reduction has nothing
// to do.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    typename Ctx,
    typename Shared>
TMOL_HD void hbond_pose_eval_derivs(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
      shared_mem_union;

  int const max_important_bond_separation = 4;

  int const pose_ind = cta / ctx.max_n_upper_triangle_inds;
  int const block_ind_pair = cta % ctx.max_n_upper_triangle_inds;

  auto upper_triangle_ind = common::upper_triangle_inds_from_linear_index(
      block_ind_pair, ctx.max_n_blocks + 1);
  int const block_ind1 = common::get<0>(upper_triangle_ind);
  int const block_ind2 = common::get<1>(upper_triangle_ind) - 1;

  int const rot_ind1 = ctx.rot_offset_for_block[pose_ind][block_ind1];
  int const rot_ind2 = ctx.rot_offset_for_block[pose_ind][block_ind2];

  if (ctx.dispatch_indices[pose_ind][block_ind1][block_ind2] == 0) {
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

  auto hbond_atom_energy =
      ([ =, &ctx ] TMOL_DEVICE_FUNC(
           int don_ind,
           int acc_ind,
           int don_tile_ind,
           int acc_tile_ind,
           int don_start,
           int acc_start,
           TMOL_THREAD HBondSingleResData<Real> const& don_dat,
           TMOL_THREAD HBondSingleResData<Real> const& acc_dat,
           TMOL_THREAD HBondRotPairData<Real> const& respair_dat,
           int cp_separation) {
        if (cp_separation < 5) {
          return Real(0.0);
        }
        int const a_ind = don_dat.block_ind;
        int const b_ind = acc_dat.block_ind;
        int const b1 = a_ind < b_ind ? a_ind : b_ind;
        int const b2 = a_ind < b_ind ? b_ind : a_ind;
        hbond_atom_derivs<TILE_SIZE, Dev, Real, Int>(
            don_ind,
            acc_ind,
            don_tile_ind,
            acc_tile_ind,
            don_start,
            acc_start,
            don_dat,
            acc_dat,
            respair_dat,
            cp_separation,
            ctx.dTdV[0][respair_dat.pose_ind][b1][b2],
            ctx.derived_coords,
            ctx.derived_atom_inds,
            ctx.pair_params,
            ctx.pair_polynomials,
            ctx.dV_dcoords);
        return Real(0.0);
      });

  auto score_inter_hbond_atom_pair = ([ =, &ctx ] SCORE_INTER_HBOND_ATOM_PAIR);

  auto score_intra_hbond_atom_pair = ([ =, &ctx ] SCORE_INTRA_HBOND_ATOM_PAIR);

  auto load_tile_invariant_interres_data =
      ([ =, &ctx ] LOAD_TILE_INVARIANT_INTERRES_DATA);

  auto load_interres1_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

  auto load_interres2_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

  auto load_interres_data_from_shared =
      ([ =, &ctx ] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([ =, &ctx ] EVAL_INTERRES_ATOM_PAIR_SCORES);

  // Derivatives accumulate as they are computed; there is nothing to store.
  auto store_calculated_energies =
      ([ =, &ctx ] TMOL_DEVICE_FUNC(
           TMOL_THREAD HBondScoringData<Real>& score_dat,
           TMOL_SHARED_REF(shared_mem_union) shared) { ; });

  auto load_tile_invariant_intrares_data =
      ([ =, &ctx ] LOAD_TILE_INVARIANT_INTRARES_DATA);

  auto load_intrares1_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

  auto load_intrares2_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

  auto load_intrares_data_from_shared =
      ([ =, &ctx ] LOAD_INTRARES_DATA_FROM_SHARED);

  auto eval_intrares_atom_pair_scores =
      ([ =, &ctx ] EVAL_INTRARES_ATOM_PAIR_SCORES);

  tmol::score::common::tile_evaluate_rot_pair<
      DeviceDispatch,
      Dev,
      HBondScoringData<Real>,
      HBondScoringData<Real>,
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

// Rotamer-pair scoring, extracted from a device lambda so that a backend
// whose entry points cannot be closures -- Metal's `kernel void` functions --
// can call the identical body.
//
// One workgroup per entry of ctx.dispatch_indices; each writes its own cell
// of ctx.output, so there are no atomics here and no reduction to make
// deterministic.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    bool AccumulateDerivs,
    typename Ctx,
    typename Shared>
TMOL_HD void hbond_rotamer_eval_energies(
    TMOL_THREAD Ctx const& ctx,
    TMOL_SHARED_REF(Shared) shared,
    int cta,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  typedef HBondSharedMem<Real, TILE_SIZE, MAX_N_CONN, launch_t>
      shared_mem_union;
  // Compile-time, not `ctx.accumulate_derivs`. The two branches of
  // HBOND_ATOM_ENERGY are both valid code, so a plain `if` on a constant is
  // enough for the optimizer to drop the unused one -- and dropping the
  // derivative branch is worth real occupancy: it takes this entry's
  // maxTotalThreadsPerThreadgroup from 384 to 448. Packing never wants
  // derivatives, so the launcher picks the energies-only instantiation for
  // every call it makes; the other exists so a gradient-carrying rotamer call
  // is still correct rather than silently wrong.
  bool const accumulate_derivs = AccumulateDerivs;

  auto hbond_atom_energy = ([ =, &ctx ] HBOND_ATOM_ENERGY);

  auto score_inter_hbond_atom_pair = ([ =, &ctx ] SCORE_INTER_HBOND_ATOM_PAIR);

  auto score_intra_hbond_atom_pair = ([ =, &ctx ] SCORE_INTRA_HBOND_ATOM_PAIR);

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
      ([ =, &ctx ] LOAD_TILE_INVARIANT_INTERRES_DATA);

  auto load_interres1_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

  auto load_interres2_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

  auto load_interres_data_from_shared =
      ([ =, &ctx ] LOAD_INTERRES_DATA_FROM_SHARED);

  auto eval_interres_atom_pair_scores =
      ([ =, &ctx ] EVAL_INTERRES_ATOM_PAIR_SCORES);

  auto store_calculated_energies =
      ([ =, &ctx ] STORE_ROTAMER_CALCULATED_ENERGIES);

  auto load_tile_invariant_intrares_data =
      ([ =, &ctx ] LOAD_TILE_INVARIANT_INTRARES_DATA);

  auto load_intrares1_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

  auto load_intrares2_tile_data_to_shared =
      ([ =, &ctx ] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

  auto load_intrares_data_from_shared =
      ([ =, &ctx ] LOAD_INTRARES_DATA_FROM_SHARED);

  auto eval_intrares_atom_pair_scores =
      ([ =, &ctx ] EVAL_INTRARES_ATOM_PAIR_SCORES);

  tmol::score::common::tile_evaluate_rot_pair<
      DeviceDispatch,
      Dev,
      HBondScoringData<Real>,
      HBondScoringData<Real>,
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
}  // namespace hbond
}  // namespace score
}  // namespace tmol
