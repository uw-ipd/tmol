
#include <tmol/utility/portable/math.hh>
#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tview_binder.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <tmol/score/hbond/identification.hh>
#include <tmol/score/ljlk/potentials/params.hh>

#include "water.hh"
#include <tmol/score/lk_ball/potentials/gen_pose_waters_ctx_tensor.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_kernels.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters_mps_launch.hh>
#include <tmol/score/lk_ball/potentials/constants.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

#define def auto TMOL_HD

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct GeneratePoseWaters {
  static auto forward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      // blocks sharing an id >= 0 move in lockstep; a water reaching
      // into one must reach into the matching rotamer
      TView<Int, 2, Dev> lockstep_group_for_block,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_don_hvy_inds,
      TView<Int, 3, Dev> block_type_tile_which_donH_for_hvy,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 3, Dev> block_type_tile_acc_n_attached_H,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      TView<LKBallWaterGenGlobalParams<Real>, 1, Dev> global_params,
      TView<Real, 1, Dev> sp2_water_tors,
      TView<Real, 1, Dev> sp3_water_tors,
      TView<Real, 1, Dev> ring_water_tors) -> TPack<Vec<Real, 3>, 2, Dev> {
    int const n_poses = first_rot_for_block.size(0);
    int const n_rots = rot_coord_offset.size(0);
    int const n_atoms = rot_coords.size(0);
    int const max_n_blocks = block_type_ind_for_rot.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_atom_all_bond_ranges.size(1);
    int const max_n_tiles = block_type_tile_n_donH.size(1);

    NVTXRange _function(__FUNCTION__);

    using tmol::score::hbond::AcceptorHybridization;

    nvtx_range_push("watergen::setup");
    auto water_coords_t =
        TPack<Vec<Real, 3>, 2, Dev>::full({n_atoms, MAX_N_WATER}, NAN);
    auto water_coords = water_coords_t.view;

    nvtx_range_pop();

    nvtx_range_push("watergen::gen");
    LAUNCH_BOX_32;
    CTA_LAUNCH_T_PARAMS;

    // State the kernel body reads, named explicitly rather than captured
    // implicitly by [=]. The lambda below closes over this one object, and
    // the Metal launch binds the same fields positionally; see
    // gen_pose_waters_ctx.hh.
    WaterGenCtx<Real, Int, Dev> ctx;
    ctx.rot_coords = rot_coords;
    ctx.rot_coord_offset = rot_coord_offset;
    ctx.pose_ind_for_rot = pose_ind_for_rot;
    ctx.block_ind_for_rot = block_ind_for_rot;
    ctx.block_type_ind_for_rot = block_type_ind_for_rot;
    ctx.first_rot_for_block = first_rot_for_block;
    ctx.first_rot_block_type = first_rot_block_type;
    ctx.lockstep_group_for_block = lockstep_group_for_block;
    ctx.rot_offset_for_block = rot_offset_for_block;
    ctx.pose_stack_inter_residue_connections =
        pose_stack_inter_residue_connections;
    ctx.block_type_n_atoms = block_type_n_atoms;
    ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
    ctx.block_type_atoms_forming_chemical_bonds =
        block_type_atoms_forming_chemical_bonds;
    ctx.block_type_n_all_bonds = block_type_n_all_bonds;
    ctx.block_type_all_bonds = block_type_all_bonds;
    ctx.block_type_atom_all_bond_ranges = block_type_atom_all_bond_ranges;
    ctx.block_type_atom_is_hydrogen = block_type_atom_is_hydrogen;
    ctx.block_type_tile_n_donH = block_type_tile_n_donH;
    ctx.block_type_tile_n_acc = block_type_tile_n_acc;
    ctx.block_type_tile_donH_inds = block_type_tile_donH_inds;
    ctx.block_type_tile_don_hvy_inds = block_type_tile_don_hvy_inds;
    ctx.block_type_tile_which_donH_for_hvy =
        block_type_tile_which_donH_for_hvy;
    ctx.block_type_tile_acc_inds = block_type_tile_acc_inds;
    ctx.block_type_tile_hybridization = block_type_tile_hybridization;
    ctx.block_type_tile_acc_n_attached_H = block_type_tile_acc_n_attached_H;
    ctx.global_params = global_params;
    ctx.sp2_water_tors = sp2_water_tors;
    ctx.sp3_water_tors = sp3_water_tors;
    ctx.ring_water_tors = ring_water_tors;
    ctx.water_coords = water_coords;
    ctx.n_rots_total = n_rots;

    auto f_watergen = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
      SHARED_MEMORY WaterGenSharedData<Real, TILE_SIZE> shared_m;
      water_gen_build_waters_for_rot<DeviceOps, Dev, Real, Int, launch_t>(
          ctx, shared_m, rot_ind);
    });

    int const n_blocks = n_poses * max_n_blocks;
    if (Dev == tmol::Device::MPS) {
      // Metal cannot launch a device lambda -- there is nowhere for a
      // closure to live, and entry points are compiled by a separate
      // toolchain -- so the same kernel body is reached through a named
      // entry. The context f_watergen closes over is the same object.
      water_gen_mps_launch_if<Dev>(ctx, n_rots);
    } else {
      DeviceOps<Dev>::template foreach_workgroup<launch_t>(
          mgr, n_rots, f_watergen);
    }

    return water_coords_t;
  };

  static auto backward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 2, Dev> dE_dWxyz,
      TView<Vec<Real, 3>, 1, Dev> rot_coords,
      TView<Int, 1, Dev> rot_coord_offset,
      TView<Int, 1, Dev> pose_ind_for_atom,
      TView<Int, 2, Dev> first_rot_for_block,
      TView<Int, 2, Dev> first_rot_block_type,
      // blocks sharing an id >= 0 move in lockstep; a water reaching
      // into one must reach into the matching rotamer
      TView<Int, 2, Dev> lockstep_group_for_block,
      TView<Int, 1, Dev> block_ind_for_rot,
      TView<Int, 1, Dev> pose_ind_for_rot,
      TView<Int, 1, Dev> block_type_ind_for_rot,
      TView<Int, 1, Dev> n_rots_for_pose,
      TView<Int, 1, Dev> rot_offset_for_pose,
      TView<Int, 2, Dev> n_rots_for_block,
      TView<Int, 2, Dev> rot_offset_for_block,
      Int max_n_rots_per_pose,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_residue_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,

      // how many inter-block chemical bonds are there
      // Dimsize: n_block_types
      TView<Int, 1, Dev> block_type_n_interblock_bonds,

      // what atoms form the inter-block chemical bonds
      // Dimsize: n_block_types x max_n_interblock_bonds
      TView<Int, 2, Dev> block_type_atoms_forming_chemical_bonds,

      TView<Int, 1, Dev> block_type_n_all_bonds,
      TView<Vec<Int, 3>, 2, Dev> block_type_all_bonds,
      TView<Vec<Int, 2>, 2, Dev> block_type_atom_all_bond_ranges,

      TView<Int, 2, Dev> block_type_tile_n_donH,
      TView<Int, 2, Dev> block_type_tile_n_acc,
      TView<Int, 3, Dev> block_type_tile_donH_inds,
      TView<Int, 3, Dev> block_type_tile_don_hvy_inds,
      TView<Int, 3, Dev> block_type_tile_which_donH_for_hvy,
      TView<Int, 3, Dev> block_type_tile_acc_inds,
      TView<Int, 3, Dev> block_type_tile_hybridization,
      TView<Int, 3, Dev> block_type_tile_acc_n_attached_H,
      TView<Int, 2, Dev> block_type_atom_is_hydrogen,

      TView<LKBallWaterGenGlobalParams<Real>, 1, Dev> global_params,
      // The three torsion tables below, packed into one [3, max_len]
      // row-major table. Built by the op wrapper and empty on every backend
      // but Metal, whose kernel context has no room for three separate
      // bindings; see WaterGenDerivCtxT.
      TView<Real, 2, Dev> packed_water_tors,
      TView<Real, 1, Dev> sp2_water_tors,
      TView<Real, 1, Dev> sp3_water_tors,
      TView<Real, 1, Dev> ring_water_tors) -> TPack<Vec<Real, 3>, 1, Dev> {
    int const n_poses = first_rot_for_block.size(0);
    int const n_rots = rot_coord_offset.size(0);
    int const n_atoms = rot_coords.size(0);
    int const max_n_blocks = block_type_ind_for_rot.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_atom_all_bond_ranges.size(1);
    int const max_n_tiles = block_type_tile_n_donH.size(1);

    NVTXRange _function(__FUNCTION__);

    using tmol::score::hbond::AcceptorHybridization;

    nvtx_range_push("watergen::setup");

    auto dE_d_pose_coords_t = TPack<Vec<Real, 3>, 1, Dev>::zeros({n_atoms});
    auto dE_d_pose_coords = dE_d_pose_coords_t.view;

    nvtx_range_pop();

    nvtx_range_push("watergen::gen");
    LAUNCH_BOX_32;
    CTA_LAUNCH_T_PARAMS;

    auto f_watergen = ([=] TMOL_DEVICE_FUNC(int rot_ind) {
      int const pose_ind = pose_ind_for_rot[rot_ind];
      int const block_type = block_type_ind_for_rot[rot_ind];
      int const block_ind = block_ind_for_rot[rot_ind];
      if (block_type == -1) {
        return;
      }

      int const n_atoms = block_type_n_atoms[block_type];

      // Allocate shared mem
      SHARED_MEMORY WaterGenSharedData<Real, TILE_SIZE> shared_m;

      // Allocate stack mem
      WaterGenData<
          tmol::score::common::TViewBinder<Dev>::template View,
          Real,
          Int>
          water_gen_dat;

      // TO DO: make this "1 body" tile action templated
      // Step 1: load in tile-invariant data for this block
      water_gen_load_tile_invariant_data<DeviceOps, Dev, nt, Int>(
          rot_coords,
          first_rot_for_block,
          first_rot_block_type,
          lockstep_group_for_block,
          rot_offset_for_block,
          rot_coord_offset,
          block_type_ind_for_rot,
          pose_stack_inter_residue_connections,

          block_type_n_all_bonds,
          block_type_all_bonds,
          block_type_atom_all_bond_ranges,
          block_type_n_interblock_bonds,
          block_type_atoms_forming_chemical_bonds,
          block_type_atom_is_hydrogen,
          global_params,

          pose_ind,
          rot_ind,
          block_ind,
          block_type,
          n_atoms,

          water_gen_dat,
          shared_m);

      // Step 2: iterate accross tiles of atoms for this block
      int const n_iterations = (n_atoms - 1) / TILE_SIZE + 1;
      for (int tile_ind = 0; tile_ind < n_iterations; ++tile_ind) {
        int const n_atoms_to_load =
            min(TILE_SIZE, n_atoms - TILE_SIZE * tile_ind);

        if (tile_ind != 0) {
          DeviceOps<Dev>::synchronize_workgroup();
        }

        // Step 3: and load data for each tile into shared memory
        water_gen_load_block_coords_and_params_into_shared<DeviceOps, Dev, nt, Int>(
            rot_coords,
            block_type_tile_n_donH,
            block_type_tile_n_acc,
            block_type_tile_donH_inds,
            block_type_tile_don_hvy_inds,
            block_type_tile_which_donH_for_hvy,
            block_type_tile_acc_inds,
            block_type_tile_hybridization,
            block_type_tile_acc_n_attached_H,
            pose_ind,
            tile_ind,
            water_gen_dat.r_dat,
            n_atoms_to_load,
            tile_ind * TILE_SIZE);
        DeviceOps<Dev>::synchronize_workgroup();

        auto dgen_tile_waters = ([&] TMOL_DEVICE_FUNC(int tid) {
          // Each iteration will build one water
          // Donor hydrogens build for their parent heavy atom and have to
          // know which hydrogen child they represent for their heavy atom
          // so they can write their computed coordinates to different
          // locations; the number of donor hydrogens must also be accounted
          // for when building waters for acceptors because some acceptors
          // are also donors
          int const n_waters = water_gen_dat.r_dat.n_donH
                               + water_gen_dat.r_dat.n_acc * MAX_N_WATER;
          for (int i = tid; i < n_waters; i += nt) {
            bool building_donor_water = i < water_gen_dat.r_dat.n_donH;
            if (building_donor_water) {
              d_build_water_for_don<TILE_SIZE, Dev>(
                  dE_dWxyz,
                  dE_d_pose_coords,
                  water_gen_dat,
                  tile_ind * TILE_SIZE,
                  i  // i is the index of the polar H within this tile
              );
            } else {
              int const acc_ind =
                  (i - water_gen_dat.r_dat.n_donH) / MAX_N_WATER;
              int const water_ind =
                  (i - water_gen_dat.r_dat.n_donH) % MAX_N_WATER;
              d_build_water_for_acc<TILE_SIZE, Dev>(
                  sp2_water_tors,
                  sp3_water_tors,
                  ring_water_tors,
                  dE_dWxyz,
                  dE_d_pose_coords,
                  water_gen_dat,
                  tile_ind * TILE_SIZE,
                  acc_ind,
                  water_ind);
            }
          }
        });
        // Step 4: ...before performing the work for each tile
        DeviceOps<Dev>::template for_each_in_workgroup<nt>(dgen_tile_waters);
      }
    });

    int const n_blocks = n_poses * max_n_blocks;
    nvtx_range_push("watergen::dgen");
    if (Dev == tmol::Device::MPS) {
      // Metal has nowhere for a closure to live, so the identical body is
      // reached through a named entry point instead.
      //
      // The three torsion tables arrive already packed into one view: the
      // backward context needs two accumulators the forward does not, and
      // Metal binds at most 31 buffers per kernel.
      int const n_sp2 = sp2_water_tors.size(0);
      int const n_sp3 = sp3_water_tors.size(0);
      int const n_ring = ring_water_tors.size(0);
      WaterGenDerivCtx<Real, Int, Dev> ctx;
      ctx.rot_coords = rot_coords;
      ctx.rot_coord_offset = rot_coord_offset;
      ctx.pose_ind_for_rot = pose_ind_for_rot;
      ctx.block_ind_for_rot = block_ind_for_rot;
      ctx.block_type_ind_for_rot = block_type_ind_for_rot;
      ctx.first_rot_for_block = first_rot_for_block;
      ctx.first_rot_block_type = first_rot_block_type;
      ctx.lockstep_group_for_block = lockstep_group_for_block;
      ctx.rot_offset_for_block = rot_offset_for_block;
      ctx.pose_stack_inter_residue_connections =
          pose_stack_inter_residue_connections;
      ctx.block_type_n_atoms = block_type_n_atoms;
      ctx.block_type_n_interblock_bonds = block_type_n_interblock_bonds;
      ctx.block_type_atoms_forming_chemical_bonds =
          block_type_atoms_forming_chemical_bonds;
      ctx.block_type_n_all_bonds = block_type_n_all_bonds;
      ctx.block_type_all_bonds = block_type_all_bonds;
      ctx.block_type_atom_all_bond_ranges = block_type_atom_all_bond_ranges;
      ctx.block_type_atom_is_hydrogen = block_type_atom_is_hydrogen;
      ctx.block_type_tile_n_donH = block_type_tile_n_donH;
      ctx.block_type_tile_n_acc = block_type_tile_n_acc;
      ctx.block_type_tile_donH_inds = block_type_tile_donH_inds;
      ctx.block_type_tile_don_hvy_inds = block_type_tile_don_hvy_inds;
      ctx.block_type_tile_which_donH_for_hvy =
          block_type_tile_which_donH_for_hvy;
      ctx.block_type_tile_acc_inds = block_type_tile_acc_inds;
      ctx.block_type_tile_hybridization = block_type_tile_hybridization;
      ctx.block_type_tile_acc_n_attached_H = block_type_tile_acc_n_attached_H;
      ctx.global_params = global_params;
      ctx.water_tors = packed_water_tors;
      ctx.dE_dWxyz = dE_dWxyz;
      ctx.dE_d_pose_coords = dE_d_pose_coords;
      ctx.n_sp2_water_tors = n_sp2;
      ctx.n_sp3_water_tors = n_sp3;
      ctx.n_ring_water_tors = n_ring;
      water_gen_derivs_mps_launch_if<Dev>(ctx, n_rots);
    } else {
      DeviceOps<Dev>::template foreach_workgroup<launch_t>(
          mgr, n_rots, f_watergen);
    }

    nvtx_range_pop();

    return dE_d_pose_coords_t;
  };
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
