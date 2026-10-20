#pragma once

// Water-generation kernel body.
//
// Separated from gen_pose_waters.impl.hh -- which holds the dispatch entry
// point and therefore reaches the torch tensor layer -- so that a Metal
// translation unit can include just the kernel. Everything named here is
// freestanding: the portable math layer, the view-generic helpers in
// water.hh, and the device-templated workgroup machinery.
//
// Holds both the forward body and the backward body: water positions depend
// on pose coordinates, so a minimization step needs both on the device.

#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/workgroup_lane.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_ctx.hh>
#include <tmol/score/lk_ball/potentials/water.hh>
// After water.hh, deliberately: constants.hh defines TILE_SIZE and
// MAX_N_WATER as macros, and those are also the names of water.hh's template
// parameters.
#include <tmol/score/lk_ball/potentials/constants.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

using tmol::score::common::WorkgroupLane;

// Build every water for one rotamer.
//
// One workgroup per rotamer; the body reads its rotamer index from the
// caller, which is foreach_workgroup on CPU and CUDA and the entry's
// threadgroup_position_in_grid on Metal.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    template <typename, int> class View,
    typename Shared>
TMOL_HD void water_gen_build_waters_for_rot(
    TMOL_THREAD WaterGenCtxT<View, Real, Int> const& ctx,
    TMOL_SHARED_REF(Shared) shared_m,
    int rot_ind,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  int const pose_ind = ctx.pose_ind_for_rot[rot_ind];
  int const block_type = ctx.block_type_ind_for_rot[rot_ind];
  int const block_ind = ctx.block_ind_for_rot[rot_ind];
  if (block_type == -1) {
    return;
  }

  int const n_atoms = ctx.block_type_n_atoms[block_type];

  // Stack-resident per-rotamer state. The threadgroup block is the
  // caller's: Metal permits `threadgroup` declarations only at kernel
  // entry scope.
  WaterGenData<View, Real, Int> water_gen_dat;

  // TO DO: make this "1 body" tile action templated
  // Step 1: load in tile-invariant data for this block
  water_gen_load_tile_invariant_data<DeviceOps, Dev, nt, Int>(
      ctx.rot_coords,
      ctx.first_rot_for_block,
      ctx.first_rot_block_type,
      ctx.lockstep_group_for_block,
      ctx.rot_offset_for_block,
      ctx.rot_coord_offset,
      ctx.block_type_ind_for_rot,
      ctx.pose_stack_inter_residue_connections,

      ctx.block_type_n_all_bonds,
      ctx.block_type_all_bonds,
      ctx.block_type_atom_all_bond_ranges,
      ctx.block_type_n_interblock_bonds,
      ctx.block_type_atoms_forming_chemical_bonds,
      ctx.block_type_atom_is_hydrogen,
      ctx.global_params,

      pose_ind,
      rot_ind,
      block_ind,
      block_type,
      n_atoms,

      water_gen_dat,
      shared_m,
      lane);

  // Step 2: iterate accross tiles of atoms for this block
  int const n_iterations = (n_atoms - 1) / TILE_SIZE + 1;
  for (int tile_ind = 0; tile_ind < n_iterations; ++tile_ind) {
    int const n_atoms_to_load = min(TILE_SIZE, n_atoms - TILE_SIZE * tile_ind);

    if (tile_ind != 0) {
      DeviceOps<Dev>::synchronize_workgroup(lane);
    }

    // Step 3: and load data for each tile into shared memory
    water_gen_load_block_coords_and_params_into_shared<DeviceOps, Dev, nt, Int>(
        ctx.rot_coords,
        ctx.block_type_tile_n_donH,
        ctx.block_type_tile_n_acc,
        ctx.block_type_tile_donH_inds,
        ctx.block_type_tile_don_hvy_inds,
        ctx.block_type_tile_which_donH_for_hvy,
        ctx.block_type_tile_acc_inds,
        ctx.block_type_tile_hybridization,
        ctx.block_type_tile_acc_n_attached_H,
        pose_ind,
        tile_ind,
        water_gen_dat.r_dat,
        n_atoms_to_load,
        tile_ind * TILE_SIZE,
        lane);
    DeviceOps<Dev>::synchronize_workgroup(lane);

    auto gen_tile_waters = ([=] TMOL_DEVICE_FUNC(int tid) {
      // Each iteration will build one water
      // Donor hydrogens build for their parent heavy atom and have to
      // know which hydrogen child they represent for their heavy atom
      // so they can write their computed coordinates to different
      // locations; the number of donor hydrogens must also be accounted
      // for when building waters for acceptors because some acceptors
      // are also donors
      int const n_waters =
          water_gen_dat.r_dat.n_donH + water_gen_dat.r_dat.n_acc * MAX_N_WATER;
      for (int i = tid; i < n_waters; i += nt) {
        bool building_donor_water = i < water_gen_dat.r_dat.n_donH;
        if (building_donor_water) {
          build_water_for_don<TILE_SIZE, Dev>(
              ctx.water_coords,
              water_gen_dat,
              tile_ind * TILE_SIZE,
              i  // i is the index of the polar H within this tile
          );
        } else {
          int const acc_ind = (i - water_gen_dat.r_dat.n_donH) / MAX_N_WATER;
          int const water_ind = (i - water_gen_dat.r_dat.n_donH) % MAX_N_WATER;
          build_water_for_acc<TILE_SIZE, Dev>(
              ctx.sp2_water_tors,
              ctx.sp3_water_tors,
              ctx.ring_water_tors,
              ctx.water_coords,
              water_gen_dat,
              tile_ind * TILE_SIZE,
              acc_ind,
              water_ind);
        }
      }
    });
    // Step 4: ...before performing the work for each tile
    DeviceOps<Dev>::template for_each_in_workgroup<nt>(gen_tile_waters, lane);
  }
}

// One row of the packed water-torsion table, presented as the 1-D table the
// water builders expect. The packed table is padded to a common width, so the
// true length has to travel alongside it; see WaterGenDerivCtxT.
template <typename RowView>
struct WaterTorsRow {
  RowView row;
  int n;
  TMOL_HD int size(int) const { return n; }
  TMOL_HD auto operator[](int i) const -> decltype(row[i]) { return row[i]; }
};

template <typename TableView>
TMOL_HD auto water_tors_row(
    TMOL_THREAD TableView const& table, int which, int n)
    -> WaterTorsRow<decltype(table[which])> {
  return WaterTorsRow<decltype(table[which])>{table[which], n};
}

// Accumulate every water derivative for one rotamer. Same traversal as the
// forward; each water scatters into dE_d_pose_coords instead of writing a
// coordinate.
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int,
    typename launch_t,
    template <typename, int> class View,
    typename Shared>
TMOL_HD void water_gen_build_water_derivs_for_rot(
    TMOL_THREAD WaterGenDerivCtxT<View, Real, Int> const& ctx,
    TMOL_SHARED_REF(Shared) shared_m,
    int rot_ind,
    WorkgroupLane<Dev> lane = {}) {
  CTA_LAUNCH_T_PARAMS;
  int const pose_ind = ctx.pose_ind_for_rot[rot_ind];
  int const block_type = ctx.block_type_ind_for_rot[rot_ind];
  int const block_ind = ctx.block_ind_for_rot[rot_ind];
  if (block_type == -1) {
    return;
  }

  int const n_atoms = ctx.block_type_n_atoms[block_type];

  // Stack-resident per-rotamer state. The threadgroup block is the
  // caller's: Metal permits `threadgroup` declarations only at kernel
  // entry scope.
  WaterGenData<View, Real, Int> water_gen_dat;

  // TO DO: make this "1 body" tile action templated
  // Step 1: load in tile-invariant data for this block
  water_gen_load_tile_invariant_data<DeviceOps, Dev, nt, Int>(
      ctx.rot_coords,
      ctx.first_rot_for_block,
      ctx.first_rot_block_type,
      ctx.lockstep_group_for_block,
      ctx.rot_offset_for_block,
      ctx.rot_coord_offset,
      ctx.block_type_ind_for_rot,
      ctx.pose_stack_inter_residue_connections,

      ctx.block_type_n_all_bonds,
      ctx.block_type_all_bonds,
      ctx.block_type_atom_all_bond_ranges,
      ctx.block_type_n_interblock_bonds,
      ctx.block_type_atoms_forming_chemical_bonds,
      ctx.block_type_atom_is_hydrogen,
      ctx.global_params,

      pose_ind,
      rot_ind,
      block_ind,
      block_type,
      n_atoms,

      water_gen_dat,
      shared_m,
      lane);

  // Step 2: iterate accross tiles of atoms for this block
  int const n_iterations = (n_atoms - 1) / TILE_SIZE + 1;
  for (int tile_ind = 0; tile_ind < n_iterations; ++tile_ind) {
    int const n_atoms_to_load = min(TILE_SIZE, n_atoms - TILE_SIZE * tile_ind);

    if (tile_ind != 0) {
      DeviceOps<Dev>::synchronize_workgroup(lane);
    }

    // Step 3: and load data for each tile into shared memory
    water_gen_load_block_coords_and_params_into_shared<DeviceOps, Dev, nt, Int>(
        ctx.rot_coords,
        ctx.block_type_tile_n_donH,
        ctx.block_type_tile_n_acc,
        ctx.block_type_tile_donH_inds,
        ctx.block_type_tile_don_hvy_inds,
        ctx.block_type_tile_which_donH_for_hvy,
        ctx.block_type_tile_acc_inds,
        ctx.block_type_tile_hybridization,
        ctx.block_type_tile_acc_n_attached_H,
        pose_ind,
        tile_ind,
        water_gen_dat.r_dat,
        n_atoms_to_load,
        tile_ind * TILE_SIZE,
        lane);
    DeviceOps<Dev>::synchronize_workgroup(lane);

    auto gen_tile_waters = ([=] TMOL_DEVICE_FUNC(int tid) {
      // Each iteration will build one water
      // Donor hydrogens build for their parent heavy atom and have to
      // know which hydrogen child they represent for their heavy atom
      // so they can write their computed coordinates to different
      // locations; the number of donor hydrogens must also be accounted
      // for when building waters for acceptors because some acceptors
      // are also donors
      int const n_waters =
          water_gen_dat.r_dat.n_donH + water_gen_dat.r_dat.n_acc * MAX_N_WATER;
      for (int i = tid; i < n_waters; i += nt) {
        bool building_donor_water = i < water_gen_dat.r_dat.n_donH;
        if (building_donor_water) {
          d_build_water_for_don<TILE_SIZE, Dev>(
              ctx.dE_dWxyz,
              ctx.dE_d_pose_coords,
              water_gen_dat,
              tile_ind * TILE_SIZE,
              i  // i is the index of the polar H within this tile
          );
        } else {
          int const acc_ind = (i - water_gen_dat.r_dat.n_donH) / MAX_N_WATER;
          int const water_ind = (i - water_gen_dat.r_dat.n_donH) % MAX_N_WATER;
          d_build_water_for_acc<TILE_SIZE, Dev>(
              water_tors_row(ctx.water_tors, 0, ctx.n_sp2_water_tors),
              water_tors_row(ctx.water_tors, 1, ctx.n_sp3_water_tors),
              water_tors_row(ctx.water_tors, 2, ctx.n_ring_water_tors),
              ctx.dE_dWxyz,
              ctx.dE_d_pose_coords,
              water_gen_dat,
              tile_ind * TILE_SIZE,
              acc_ind,
              water_ind);
        }
      }
    });
    // Step 4: ...before performing the work for each tile
    DeviceOps<Dev>::template for_each_in_workgroup<nt>(gen_tile_waters, lane);
  }
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
