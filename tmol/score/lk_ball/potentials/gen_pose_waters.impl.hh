#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include <tmol/score/hbond/identification.hh>
#include <tmol/score/ljlk/potentials/params.hh>

#include "water.hh"
#include <tmol/score/lk_ball/potentials/constants.hh>

#include <iostream>  // TEMP!

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct GeneratePoseWaters {
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> pose_coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

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
      TView<Real, 1, Dev> ring_water_tors) -> TPack<Vec<Real, 3>, 3, Dev> {
    int const n_poses = pose_coords.size(0);
    int const max_n_pose_atoms = pose_coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_atom_all_bond_ranges.size(1);
    int const max_n_tiles = block_type_tile_n_donH.size(1);

    assert(pose_stack_block_coord_offset.size(0) == n_poses);
    assert(pose_stack_block_type.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);
    assert(block_type_n_interblock_bonds.size(0) == n_block_types);
    assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
    assert(block_type_atoms_forming_chemical_bonds.size(1) == max_n_conn);
    assert(block_type_n_all_bonds.size(0) == n_block_types);
    assert(block_type_atom_all_bond_ranges.size(0) == n_block_types);
    assert(block_type_tile_n_donH.size(0) == n_block_types);
    assert(block_type_tile_n_acc.size(0) == n_block_types);
    assert(block_type_tile_n_acc.size(1) == max_n_tiles);
    assert(block_type_tile_donH_inds.size(0) == n_block_types);
    assert(block_type_tile_donH_inds.size(1) == max_n_tiles);
    assert(block_type_tile_donH_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_don_hvy_inds.size(0) == n_block_types);
    assert(block_type_tile_don_hvy_inds.size(1) == max_n_tiles);
    assert(block_type_tile_don_hvy_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_which_donH_for_hvy.size(0) == n_block_types);
    assert(block_type_tile_which_donH_for_hvy.size(1) == max_n_tiles);
    assert(block_type_tile_which_donH_for_hvy.size(2) == TILE_SIZE);
    assert(block_type_tile_acc_inds.size(0) == n_block_types);
    assert(block_type_tile_acc_inds.size(1) == max_n_tiles);
    assert(block_type_tile_acc_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_hybridization.size(0) == n_block_types);
    assert(block_type_tile_hybridization.size(1) == max_n_tiles);
    assert(block_type_tile_hybridization.size(2) == TILE_SIZE);
    assert(block_type_tile_acc_n_attached_H.size(0) == n_block_types);
    assert(block_type_tile_acc_n_attached_H.size(1) == max_n_tiles);
    assert(block_type_tile_acc_n_attached_H.size(2) == TILE_SIZE);
    assert(block_type_atom_is_hydrogen.size(0) == n_block_types);
    assert(block_type_atom_is_hydrogen.size(1) == max_n_block_atoms);

    NVTXRange _function(__FUNCTION__);

    using tmol::score::hbond::AcceptorBases;
    using tmol::score::hbond::AcceptorHybridization;

    nvtx_range_push("watergen::setup");
    auto water_coords_t = TPack<Vec<Real, 3>, 3, Dev>::full(
        {n_poses, max_n_pose_atoms, MAX_N_WATER}, NAN);
    auto water_coords = water_coords_t.view;

    nvtx_range_pop();

    nvtx_range_push("watergen::gen");
    LAUNCH_BOX_32;
    CTA_LAUNCH_T_PARAMS;

    auto f_watergen = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose_ind = ind / max_n_blocks;
      int const block_ind = ind % max_n_blocks;
      int const block_type = pose_stack_block_type[pose_ind][block_ind];

      if (block_type < 0) return;

      int const n_atoms = block_type_n_atoms[block_type];

      // Allocate shared mem
      SHARED_MEMORY WaterGenSharedData<Real, TILE_SIZE> shared_m;

      // Allocate stack mem
      WaterGenData<Dev, Real, Int> water_gen_dat;

      // TO DO: make this "1 body" tile action templated
      // Step 1: load in tile-invariant data for this block
      water_gen_load_tile_invariant_data<DeviceOps, Dev, nt>(
          pose_coords,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          pose_stack_inter_residue_connections,

          block_type_n_all_bonds,
          block_type_all_bonds,
          block_type_atom_all_bond_ranges,
          block_type_n_interblock_bonds,
          block_type_atoms_forming_chemical_bonds,
          block_type_atom_is_hydrogen,
          global_params,

          pose_ind,
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
        water_gen_load_block_coords_and_params_into_shared<DeviceOps, Dev, nt>(
            pose_coords,
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

        auto gen_tile_waters = ([&] TMOL_DEVICE_FUNC(int tid) {
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
              build_water_for_don<TILE_SIZE>(
                  water_coords,
                  water_gen_dat,
                  tile_ind * TILE_SIZE,
                  i  // i is the index of the polar H within this tile
              );
            } else {
              int const acc_ind =
                  (i - water_gen_dat.r_dat.n_donH) / MAX_N_WATER;
              int const water_ind =
                  (i - water_gen_dat.r_dat.n_donH) % MAX_N_WATER;
              build_water_for_acc<TILE_SIZE>(
                  sp2_water_tors,
                  sp3_water_tors,
                  ring_water_tors,
                  water_coords,
                  water_gen_dat,
                  tile_ind * TILE_SIZE,
                  acc_ind,
                  water_ind);
            }
          }
        });
        // Step 4: ...before performing the work for each tile
        DeviceOps<Dev>::template for_each_in_workgroup<nt>(gen_tile_waters);
      }
    });

    int const n_blocks = n_poses * max_n_blocks;
    DeviceOps<Dev>::template foreach_workgroup<launch_t>(n_blocks, f_watergen);

    return water_coords_t;
  };

  static auto backward(
      TView<Vec<Real, 3>, 3, Dev> dE_dWxyz,
      TView<Vec<Real, 3>, 2, Dev> pose_coords,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

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
    int const n_poses = pose_coords.size(0);
    int const max_n_pose_atoms = pose_coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_atom_all_bond_ranges.size(1);
    int const max_n_tiles = block_type_tile_n_donH.size(1);

    assert(pose_stack_block_coord_offset.size(0) == n_poses);
    assert(pose_stack_block_type.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);
    assert(block_type_n_interblock_bonds.size(0) == n_block_types);
    assert(block_type_atoms_forming_chemical_bonds.size(0) == n_block_types);
    assert(block_type_atoms_forming_chemical_bonds.size(1) == max_n_conn);
    assert(block_type_n_all_bonds.size(0) == n_block_types);
    assert(block_type_atom_all_bond_ranges.size(0) == n_block_types);
    assert(block_type_tile_n_donH.size(0) == n_block_types);
    assert(block_type_tile_n_acc.size(0) == n_block_types);
    assert(block_type_tile_n_acc.size(1) == max_n_tiles);
    assert(block_type_tile_donH_inds.size(0) == n_block_types);
    assert(block_type_tile_donH_inds.size(1) == max_n_tiles);
    assert(block_type_tile_donH_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_don_hvy_inds.size(0) == n_block_types);
    assert(block_type_tile_don_hvy_inds.size(1) == max_n_tiles);
    assert(block_type_tile_don_hvy_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_which_donH_for_hvy.size(0) == n_block_types);
    assert(block_type_tile_which_donH_for_hvy.size(1) == max_n_tiles);
    assert(block_type_tile_which_donH_for_hvy.size(2) == TILE_SIZE);
    assert(block_type_tile_acc_inds.size(0) == n_block_types);
    assert(block_type_tile_acc_inds.size(1) == max_n_tiles);
    assert(block_type_tile_acc_inds.size(2) == TILE_SIZE);
    assert(block_type_tile_hybridization.size(0) == n_block_types);
    assert(block_type_tile_hybridization.size(1) == max_n_tiles);
    assert(block_type_tile_hybridization.size(2) == TILE_SIZE);
    assert(block_type_tile_acc_n_attached_H.size(0) == n_block_types);
    assert(block_type_tile_acc_n_attached_H.size(1) == max_n_tiles);
    assert(block_type_tile_acc_n_attached_H.size(2) == TILE_SIZE);
    assert(block_type_atom_is_hydrogen.size(0) == n_block_types);
    assert(block_type_atom_is_hydrogen.size(1) == max_n_block_atoms);

    // std::cout << "d watergen start" << std::endl;

    NVTXRange _function(__FUNCTION__);

    nvtx_range_push("watergen::dsetup");

    using tmol::score::hbond::AcceptorBases;
    using tmol::score::hbond::AcceptorHybridization;

    auto dE_d_pose_coords_t =
        TPack<Vec<Real, 3>, 2, Dev>::zeros({n_poses, max_n_pose_atoms});
    auto dE_d_pose_coords = dE_d_pose_coords_t.view;

    int nsp2wats = sp2_water_tors.size(0);
    int nsp3wats = sp3_water_tors.size(0);
    int nringwats = ring_water_tors.size(0);
    nvtx_range_pop();

    LAUNCH_BOX_32;
    CTA_LAUNCH_T_PARAMS;

    auto f_watergen = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose_ind = ind / max_n_blocks;
      int const block_ind = ind % max_n_blocks;
      int const block_type = pose_stack_block_type[pose_ind][block_ind];

      if (block_type < 0) return;

      int const n_atoms = block_type_n_atoms[block_type];

      // Allocate shared mem
      SHARED_MEMORY WaterGenSharedData<Real, TILE_SIZE> shared_m;

      // Allocate stack mem
      WaterGenData<Dev, Real, Int> water_gen_dat;

      // TO DO: make this "1 body" tile action templated
      // Step 1: load in tile-invariant data for this block
      // printf("water_gen_load_tile_invariant_data %d\n", ind);
      water_gen_load_tile_invariant_data<DeviceOps, Dev, nt>(
          pose_coords,
          pose_stack_block_coord_offset,
          pose_stack_block_type,
          pose_stack_inter_residue_connections,

          block_type_n_all_bonds,
          block_type_all_bonds,
          block_type_atom_all_bond_ranges,
          block_type_n_interblock_bonds,
          block_type_atoms_forming_chemical_bonds,
          block_type_atom_is_hydrogen,
          global_params,

          pose_ind,
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
        // printf("water_gen_load_block_coords_and_params_into_shared %d %d\n",
        // ind, tile_ind);
        water_gen_load_block_coords_and_params_into_shared<DeviceOps, Dev, nt>(
            pose_coords,
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
            // printf("dgen_tile_waters %d %d %d\n", ind, tile_ind, tid);
            bool building_donor_water = i < water_gen_dat.r_dat.n_donH;
            if (building_donor_water) {
              d_build_water_for_don<TILE_SIZE>(
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
              d_build_water_for_acc<TILE_SIZE>(
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
    DeviceOps<Dev>::template foreach_workgroup<launch_t>(n_blocks, f_watergen);
    nvtx_range_pop();

    // std::cout << "d watergen end" << std::endl;

    return dE_d_pose_coords_t;
  };
};

#undef def

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
