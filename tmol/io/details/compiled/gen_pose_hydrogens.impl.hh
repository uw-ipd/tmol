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
#include <tmol/score/common/uaid_util.hh>

#include <tmol/io/details/compiled/gen_coord.hh>

// #include <iostream>  // TEMP!

namespace tmol {
namespace io {
namespace details {
namespace compiled {

#define def auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct GeneratePoseHydrogens {
  static auto forward(
      TView<Vec<Real, 3>, 2, Dev> pose_coords,
      TView<Int, 2, Dev> h_coords_missing,
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
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      TView<Int, 4, Dev> block_type_atom_ancestors,  // n-bt x max-n-ats x 4 x 3

      TView<Real, 3, Dev>
          block_type_atom_icoors  // n-bt x max-n-ats x 3 [phi, theta, D]

      ) -> TPack<Vec<Real, 3>, 2, Dev> {
    int const n_poses = pose_coords.size(0);
    int const max_n_pose_atoms = pose_coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_residue_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_atom_downstream_of_conn.size(2);

    assert(h_coords_missing.size(0) == n_poses);
    assert(h_coords_missing.size(1) == max_n_pose_atoms);
    assert(pose_stack_block_coord_offset.size(0) == n_poses);
    assert(pose_stack_block_type.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);
    assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
    assert(block_type_atom_downstream_of_conn.size(1) == max_n_conn);
    assert(block_type_atom_ancestors.size(0) == n_block_types);
    assert(block_type_atom_ancestors.size(1) == max_n_block_atoms);
    assert(block_type_atom_ancestors.size(2) == 4);
    assert(block_type_atom_ancesotrs.size(3) == 3);  // uaid = 3 indices
    assert(block_type_atom_icoors.size(0) == n_block_types);
    assert(block_type_atom_icoors.size(1) == max_n_block_atoms);
    assert(block_type_atom_icoors.size(2) == 3);  // phi, theta, D

    auto new_coords_t =
        TPack<Real, 3, D>::zeros({n_poses, max_n_pose_atoms, 3});
    auto new_coords = new_coords_t.view;

    LAUNCH_BOX_32;
    CTA_LAUNCH_T_PARAMS;

    auto f_coord_builder = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose_ind = ind / (max_n_blocks * max_n_block_atoms);
      ind = ind - (pose_ind * max_n_blocks * max_n_block_atoms);
      int const block_ind = ind / max_n_block_atoms;
      int const atom_ind = ind % max_n_block_atoms;
      int const block_type = pose_block_type[pose_ind][block_ind];
      if (block_type < 0) {
        return;
      }
      int const n_block_type_atoms = block_type_n_atoms[block_type];
      if (atom_ind >= n_block_type_atoms) {
        return;
      }
      int const block_offset =
          pose_stack_block_coord_offset[pose_ind][block_ind];

      // is this a hydrogen we should build coordinates for?
      if (h_coord_and_missing[pose_ind][block_offset + atom_ind]) {
        // ok! build the coordinate
        auto get_anc = ([&] TMOL_DEVICE_FUNC(int which_anc) {
          return resolve_atom_from_uaid<Real, Int, Dev>(
              block_type_atom_ancestors[block_type][atom_ind][which_anc],
              block_ind,
              pose_ind,
              pose_stack_block_coord_offset,
              pose_stack_block_type,
              pose_stack_inter_block_connections,
              block_type_atom_downstream_of_conn)
        });
        int anc0 = get_anc(0);
        int anc1 = get_anc(1);
        int anc2 = get_anc(2);
        if (anc0 == -1 || anc1 == -1 || anc2 == -1) {
          // cannot build a hydrogen if we're missing the ancestors
          return;
        }

        Vec<Real, 3> coord0 = load_coord(anc0);
        Vec<Real, 3> coord1 = load_coord(anc1);
        Vec<Real, 3> coord2 = load_coord(anc2);

        Vec<Real, 3> new_coord = build_coordinate.V(
            coord0,
            coord1,
            coord2,
            block_type_atom_icoors[block_type][atom_ind][2],   // D
            block_type_atom_icoors[block_type][atom_ind][1],   // theta
            block_type_atom_icoors[block_type][atom_ind][0]);  // phi
        new_coords[pose_ind][block_offset + atom_ind] = new_coord;
      } else {
        // copy the coordinate from the input coordinates
        Vec<Real, 3> atom_coord =
            pose_coords[pose_ind][block_offset + atom_ind];
        new_pose_coords[pose_ind][block_offset + atom_ind] = atom_coord;
      }
    });

    int const n_atoms = n_poses * max_n_blocks * max_n_block_atoms;
    ;
    DeviceOps<Dev>::template foreach_workgroup<launch_t>(
        n_atoms, f_coord_builder);

    return new_coords_t;
  };

  static auto backward(
      TView<Vec<Real, 3>, 3, Dev> dE_d_new_coords,
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
    int const max_n_block_atoms = block_type_atom_downstream_of_conn.size(2);

    assert(h_coords_missing.size(0) == n_poses);
    assert(h_coords_missing.size(1) == max_n_pose_atoms);
    assert(pose_stack_block_coord_offset.size(0) == n_poses);
    assert(pose_stack_block_type.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(0) == n_poses);
    assert(pose_stack_inter_residue_connections.size(1) == max_n_blocks);
    assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
    assert(block_type_atom_downstream_of_conn.size(1) == max_n_conn);
    assert(block_type_atom_ancestors.size(0) == n_block_types);
    assert(block_type_atom_ancestors.size(1) == max_n_block_atoms);
    assert(block_type_atom_ancestors.size(2) == 4);
    assert(block_type_atom_ancesotrs.size(3) == 3);  // uaid = 3 indices
    assert(block_type_atom_icoors.size(0) == n_block_types);
    assert(block_type_atom_icoors.size(1) == max_n_block_atoms);
    assert(block_type_atom_icoors.size(2) == 3);  // phi, theta, D

    auto dE_d_orig_coords_t =
        TPack<Real, 3, D>::zeros({n_poses, max_n_pose_atoms, 3});
    auto dE_d_orig_coords = dE_d_orig_coords_t.view;

    auto f_coord_builder_derivs = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose_ind = ind / (max_n_blocks * max_n_block_atoms);
      ind = ind - (pose_ind * max_n_blocks * max_n_block_atoms);
      int const block_ind = ind / max_n_block_atoms;
      int const atom_ind = ind % max_n_block_atoms;
      int const block_type = pose_block_type[pose_ind][block_ind];
      if (block_type < 0) {
        return;
      }
      int const n_block_type_atoms = block_type_n_atoms[block_type];
      if (atom_ind >= n_block_type_atoms) {
        return;
      }
      int const block_offset =
          pose_stack_block_coord_offset[pose_ind][block_ind];

      // is this a hydrogen we should build coordinates for?
      if (h_coord_and_missing[pose_ind][block_offset + atom_ind]) {
        // ok! build the coordinate
        auto get_anc = ([&] TMOL_DEVICE_FUNC(int which_anc) {
          return resolve_atom_from_uaid<Real, Int, Dev>(
              block_type_atom_ancestors[block_type][atom_ind][which_anc],
              block_ind,
              pose_ind,
              pose_stack_block_coord_offset,
              pose_stack_block_type,
              pose_stack_inter_block_connections,
              block_type_atom_downstream_of_conn)
        });
        int const anc0 = get_anc(0);
        int const anc1 = get_anc(1);
        int const anc2 = get_anc(2);
        if (anc0 == -1 || anc1 == -1 || anc2 == -1) {
          // cannot build a hydrogen if we're missing the ancestors
          return;
        }

        Vec<Real, 3> coord0 = load_coord(anc0);
        Vec<Real, 3> coord1 = load_coord(anc1);
        Vec<Real, 3> coord2 = load_coord(anc2);

        auto coord_derivs = build_coordinate.dV(
            coord0,
            coord1,
            coord2,
            block_type_atom_icoors[block_type][atom_ind][2],   // D
            block_type_atom_icoors[block_type][atom_ind][1],   // theta
            block_type_atom_icoors[block_type][atom_ind][0]);  // phi

        Vec<Real, 3> dE_dH = dE_d_new_coords[pose_ind][block_offset + atom_ind];

        // new_coords[pose_ind][block_offset + atom_ind] = new_coord;
        common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][anc0], coord_derivs.dp * dE_dH);
        common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][anc1], coord_derivs.dgp * dE_dH);
        common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][anc2], coord_derivs.dggp * dE_dH);

      } else {
        // "copy" the coordinate from the input coordinates
        Vec<Real, 3> atom_deriv =
            dE_d_new_coords[pose_ind][block_offset + atom_ind];
        common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][block_offset + atom_ind], atom_deriv);
      }
    });

    int const n_atoms = n_poses * max_n_blocks * max_n_block_atoms;
    ;
    DeviceOps<Dev>::template foreach_workgroup<launch_t>(
        n_atoms, f_coord_builder_derivs);

    return dE_d_orig_coords_t;
  };
};

#undef def

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
