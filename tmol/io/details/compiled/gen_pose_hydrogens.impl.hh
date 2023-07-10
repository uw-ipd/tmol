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
      TView<Vec<Real, 3>, 2, Dev> orig_coords,
      TView<Int, 3, Dev> h_atom_missing,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // n-bt x max-n-ats x 3 x 3
      TView<Int, 4, Dev> block_type_atom_ancestors,

      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors,

      // TEMP! Handle the case when an atom's coordinate depends on
      // an un-resolvable atom, e.g., "down" for an N-terminal atom
      // n-bt x max-n-ats x 3 x 3
      TView<Int, 4, Dev> block_type_atom_ancestors_backup,
      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors_backup

      ) -> TPack<Vec<Real, 3>, 2, Dev> {
    int const n_poses = orig_coords.size(0);
    int const max_n_pose_atoms = orig_coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_block_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_atom_downstream_of_conn.size(2);

    assert(h_atom_missing.size(0) == n_poses);
    assert(h_atom_missing.size(1) == max_n_blocks);
    assert(h_atom_missing.size(2) == max_n_block_atoms);
    assert(pose_stack_block_coord_offset.size(0) == n_poses);
    assert(pose_stack_block_type.size(0) == n_poses);
    assert(pose_stack_inter_block_connections.size(0) == n_poses);
    assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
    assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
    assert(block_type_atom_downstream_of_conn.size(1) == max_n_conn);
    assert(block_type_atom_ancestors.size(0) == n_block_types);
    assert(block_type_atom_ancestors.size(1) == max_n_block_atoms);
    assert(
        block_type_atom_ancestors.size(2)
        == 3);  // parent, grandparent, great grandparent
    assert(block_type_atom_ancestors.size(3) == 3);  // uaid = 3 indices
    assert(block_type_atom_icoors.size(0) == n_block_types);
    assert(block_type_atom_icoors.size(1) == max_n_block_atoms);
    assert(block_type_atom_icoors.size(2) == 3);  // phi, theta, D

    auto new_coords_t =
        TPack<Vec<Real, 3>, 2, Dev>::zeros({n_poses, max_n_pose_atoms});
    auto new_coords = new_coords_t.view;

    LAUNCH_BOX_32;

    auto f_coord_builder = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose_ind = ind / (max_n_blocks * max_n_block_atoms);
      ind = ind - (pose_ind * max_n_blocks * max_n_block_atoms);
      int const block_ind = ind / max_n_block_atoms;
      int const atom_ind = ind % max_n_block_atoms;
      int const block_type = pose_stack_block_type[pose_ind][block_ind];
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
      if (h_atom_missing[pose_ind][block_ind][atom_ind]) {
        // ok! build the coordinate
        auto get_anc = ([&] TMOL_DEVICE_FUNC(int which_anc) {
          return score::common::resolve_atom_from_uaid<Int, Dev>(
              block_type_atom_ancestors[block_type][atom_ind][which_anc],
              block_ind,
              pose_ind,
              pose_stack_block_coord_offset,
              pose_stack_block_type,
              pose_stack_inter_block_connections,
              block_type_atom_downstream_of_conn);
        });
        int anc0, anc1, anc2;
        float D, theta, phi;

        anc0 = get_anc(0);
        anc1 = get_anc(1);
        anc2 = get_anc(2);
        if (anc0 == -1 || anc1 == -1 || anc2 == -1) {
          auto get_backup_anc = ([&] TMOL_DEVICE_FUNC(int which_anc) {
            return score::common::resolve_atom_from_uaid<Int, Dev>(
                block_type_atom_ancestors_backup[block_type][atom_ind]
                                                [which_anc],
                block_ind,
                pose_ind,
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
          });
          anc0 = get_backup_anc(0);
          anc1 = get_backup_anc(1);
          anc2 = get_backup_anc(2);
          // printf("fell back on ancestors %d %d and %d when building atom %d
          // on res %d\n", anc0, anc1, anc2, atom_ind, block_ind);
          if (anc0 == -1 || anc1 == -1 || anc2 == -1) {
            // cannot build a hydrogen if we're missing the ancestors
            return;
          }
          D = block_type_atom_icoors_backup[block_type][atom_ind][2];  // D
          theta =
              block_type_atom_icoors_backup[block_type][atom_ind][1];  // theta
          phi = block_type_atom_icoors_backup[block_type][atom_ind][0];  // phi

        } else {
          D = block_type_atom_icoors[block_type][atom_ind][2];      // D
          theta = block_type_atom_icoors[block_type][atom_ind][1];  // theta
          phi = block_type_atom_icoors[block_type][atom_ind][0];    // phi
        }

        // printf("res %d atom %d, ancestors %d %d and %d\n", block_ind,
        // atom_ind, anc0, anc1, anc2);
        Vec<Real, 3> coord0 = orig_coords[pose_ind][anc0];
        Vec<Real, 3> coord1 = orig_coords[pose_ind][anc1];
        Vec<Real, 3> coord2 = orig_coords[pose_ind][anc2];

        Vec<Real, 3> new_coord =
            build_coordinate<Real>::V(coord0, coord1, coord2, D, theta, phi);
        // printf(
        //     "new coord %d %d %d (%f %f %f), (%f %f %f)\n",
        //     block_type,
        //     block_ind,
        //     atom_ind,
        //     D,
        //     theta,
        //     phi,
        //     new_coord[0],
        //     new_coord[1] * 180 / 3.14159,
        //     new_coord[2]);

        new_coords[pose_ind][block_offset + atom_ind] = new_coord;
      } else {
        // copy the coordinate from the input coordinates
        Vec<Real, 3> atom_coord =
            orig_coords[pose_ind][block_offset + atom_ind];
        new_coords[pose_ind][block_offset + atom_ind] = atom_coord;
      }
    });

    int const n_atoms = n_poses * max_n_blocks * max_n_block_atoms;
    DeviceOps<Dev>::template forall<launch_t>(n_atoms, f_coord_builder);

    return new_coords_t;
  };

  static auto backward(
      TView<Vec<Real, 3>, 2, Dev> dE_d_new_coords,
      TView<Vec<Real, 3>, 2, Dev> new_coords,
      TView<Vec<Real, 3>, 2, Dev> orig_coords,
      TView<Int, 3, Dev> h_atom_missing,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,

      // For determining which atoms to retrieve from neighboring
      // residues we have to know how the blocks in the Pose
      // are connected
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,

      //////////////////////
      // Chemical properties
      // how many atoms for a given block
      // Dimsize n_block_types
      TView<Int, 1, Dev> block_type_n_atoms,
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

      // n-bt x max-n-ats x 3 x 3
      TView<Int, 4, Dev> block_type_atom_ancestors,

      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors,

      // TEMP! Handle the case when an atom's coordinate depends on
      // an un-resolvable atom, e.g., "down" for an N-terminal atom
      // n-bt x max-n-ats x 3 x 3
      TView<Int, 4, Dev> block_type_atom_ancestors_backup,
      // n-bt x max-n-ats x 3 [phi, theta, D]
      TView<Real, 3, Dev> block_type_atom_icoors_backup

      ) -> TPack<Vec<Real, 3>, 2, Dev> {
    int const n_poses = orig_coords.size(0);
    int const max_n_pose_atoms = orig_coords.size(1);
    int const max_n_blocks = pose_stack_block_type.size(1);
    int const max_n_conn = pose_stack_inter_block_connections.size(2);
    int const n_block_types = block_type_n_atoms.size(0);
    int const max_n_block_atoms = block_type_atom_downstream_of_conn.size(2);

    // printf(
    //     "dE_d_new_coords: %d %d\n",
    //     dE_d_new_coords.size(0),
    //     dE_d_new_coords.size(1));

    assert(h_atom_missing.size(0) == n_poses);
    assert(h_atom_missing.size(1) == max_n_blocks);
    assert(h_atom_missing.size(2) == max_n_block_atoms);
    assert(pose_stack_block_coord_offset.size(0) == n_poses);
    assert(pose_stack_block_type.size(0) == n_poses);
    assert(pose_stack_inter_block_connections.size(0) == n_poses);
    assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
    assert(block_type_atom_downstream_of_conn.size(0) == n_block_types);
    assert(block_type_atom_downstream_of_conn.size(1) == max_n_conn);
    assert(block_type_atom_ancestors.size(0) == n_block_types);
    assert(block_type_atom_ancestors.size(1) == max_n_block_atoms);
    assert(
        block_type_atom_ancestors.size(2)
        == 3);  // parent, grandparent, great grandparent
    assert(block_type_atom_ancestors.size(3) == 3);  // uaid = 3 indices
    assert(block_type_atom_icoors.size(0) == n_block_types);
    assert(block_type_atom_icoors.size(1) == max_n_block_atoms);
    assert(block_type_atom_icoors.size(2) == 3);  // phi, theta, D

    auto dE_d_orig_coords_t =
        TPack<Vec<Real, 3>, 2, Dev>::zeros({n_poses, max_n_pose_atoms});
    auto dE_d_orig_coords = dE_d_orig_coords_t.view;

    LAUNCH_BOX_32;

    auto f_coord_builder_derivs = ([=] TMOL_DEVICE_FUNC(int ind) {
      int const pose_ind = ind / (max_n_blocks * max_n_block_atoms);
      ind = ind - (pose_ind * max_n_blocks * max_n_block_atoms);
      int const block_ind = ind / max_n_block_atoms;
      int const atom_ind = ind % max_n_block_atoms;
      int const block_type = pose_stack_block_type[pose_ind][block_ind];
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
      if (h_atom_missing[pose_ind][block_ind][atom_ind]) {
        // ok! build the coordinate
        auto get_anc = ([&] TMOL_DEVICE_FUNC(int which_anc) {
          return score::common::resolve_atom_from_uaid<Int, Dev>(
              block_type_atom_ancestors[block_type][atom_ind][which_anc],
              block_ind,
              pose_ind,
              pose_stack_block_coord_offset,
              pose_stack_block_type,
              pose_stack_inter_block_connections,
              block_type_atom_downstream_of_conn);
        });
        int anc0, anc1, anc2;
        float D, theta, phi;
        anc0 = get_anc(0);
        anc1 = get_anc(1);
        anc2 = get_anc(2);
        if (anc0 == -1 || anc1 == -1 || anc2 == -1) {
          auto get_backup_anc = ([&] TMOL_DEVICE_FUNC(int which_anc) {
            return score::common::resolve_atom_from_uaid<Int, Dev>(
                block_type_atom_ancestors_backup[block_type][atom_ind]
                                                [which_anc],
                block_ind,
                pose_ind,
                pose_stack_block_coord_offset,
                pose_stack_block_type,
                pose_stack_inter_block_connections,
                block_type_atom_downstream_of_conn);
          });
          anc0 = get_backup_anc(0);
          anc1 = get_backup_anc(1);
          anc2 = get_backup_anc(2);
          // printf("fell back on ancestors %d %d and %d when building atom %d
          // on res %d\n", anc0, anc1, anc2, atom_ind, block_ind);
          if (anc0 == -1 || anc1 == -1 || anc2 == -1) {
            // cannot build a hydrogen if we're missing the ancestors
            return;
          }

          D = block_type_atom_icoors_backup[block_type][atom_ind][2];  // D
          theta =
              block_type_atom_icoors_backup[block_type][atom_ind][1];  // theta
          phi = block_type_atom_icoors_backup[block_type][atom_ind][0];  // phi

        } else {
          D = block_type_atom_icoors[block_type][atom_ind][2];      // D
          theta = block_type_atom_icoors[block_type][atom_ind][1];  // theta
          phi = block_type_atom_icoors[block_type][atom_ind][0];    // phi
        }

        // printf("res %d atom %d, ancestors %d %d and %d\n", block_ind,
        // atom_ind, anc0, anc1, anc2);
        Vec<Real, 3> coord0 = orig_coords[pose_ind][anc0];
        Vec<Real, 3> coord1 = orig_coords[pose_ind][anc1];
        Vec<Real, 3> coord2 = orig_coords[pose_ind][anc2];

        auto coord_derivs =
            build_coordinate<Real>::dV(coord0, coord1, coord2, D, theta, phi);

        Vec<Real, 3> dE_dH = dE_d_new_coords[pose_ind][block_offset + atom_ind];

        // printf(
        //     "new coord %d %d (%f %f %f)\n",
        //     atom_ind,
        //     block_offset,
        //     new_coords[pose_ind][block_offset + atom_ind][0],
        //     new_coords[pose_ind][block_offset + atom_ind][1],
        //     new_coords[pose_ind][block_offset + atom_ind][2]);

        // printf(
        //     "sum deriv %d %d %d (%f %f %f), (%f %f %f)\n",
        //     atom_ind,
        //     block_offset,
        //     anc0,
        //     dE_dH(0),
        //     dE_dH(1),
        //     dE_dH(2),
        //     (coord_derivs.dp * dE_dH)[0],
        //     (coord_derivs.dp * dE_dH)[1],
        //     (coord_derivs.dp * dE_dH)[2]);
        // printf(
        //     "sum deriv %d %d %d (%f %f %f), (%f %f %f)\n",
        //     atom_ind,
        //     block_offset,
        //     anc1,
        //     dE_dH(0),
        //     dE_dH(1),
        //     dE_dH(2),
        //     (coord_derivs.dgp * dE_dH)[0],
        //     (coord_derivs.dgp * dE_dH)[1],
        //     (coord_derivs.dgp * dE_dH)[2]);
        // printf(
        //     "sum deriv %d %d %d (%f %f %f), (%f %f %f)\n",
        //     atom_ind,
        //     block_offset,
        //     anc2,
        //     dE_dH(0),
        //     dE_dH(1),
        //     dE_dH(2),
        //     (coord_derivs.dggp * dE_dH)[0],
        //     (coord_derivs.dggp * dE_dH)[1],
        //     (coord_derivs.dggp * dE_dH)[2]);
        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][anc0], coord_derivs.dp * dE_dH);
        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][anc1], coord_derivs.dgp * dE_dH);
        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][anc2], coord_derivs.dggp * dE_dH);

      } else {
        // "copy" the derivative from the input derivatives; except, since we
        // may be accumulating into the derivative the derivative wrt hydrogen
        // atoms that depend on the coordinate of this atom, then we have to
        // perform an atomic add
        Vec<Real, 3> atom_deriv =
            dE_d_new_coords[pose_ind][block_offset + atom_ind];
        if (atom_deriv[0] != atom_deriv[0] || atom_deriv[1] != atom_deriv[1]
            || atom_deriv[2] != atom_deriv[2]) {
          printf(
              "copy deriv NaN !!! %d %d %d (%f %f %f)\n",
              pose_ind,
              block_ind,
              atom_ind,
              atom_deriv(0),
              atom_deriv(1),
              atom_deriv(2));
        }
        score::common::accumulate<Dev, Vec<Real, 3>>::add(
            dE_d_orig_coords[pose_ind][block_offset + atom_ind], atom_deriv);
      }
    });

    int const n_atoms = n_poses * max_n_blocks * max_n_block_atoms;
    DeviceOps<Dev>::template forall<launch_t>(n_atoms, f_coord_builder_derivs);

    return dE_d_orig_coords_t;
  };
};

#undef def

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
