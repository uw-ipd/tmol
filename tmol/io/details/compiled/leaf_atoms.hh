#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/score/common/uaid_util.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Int, typename Real, tmol::Device Dev>
struct select_leaf_ancestors {
  struct anc_and_geom {
    Int anc0;
    Int anc1;
    Int anc2;
    Real D;
    Real theta;
    Real phi;

    static anc_and_geom EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE empty() {
      return {Int(-1), Int(-1), Int(-1), Real(0), Real(0), Real(0)};
    }
  };

  static auto EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE find_ancestors(
      int pose_ind,
      int block_ind,
      int atom_ind,
      int block_type,
      TView<Int, 2, Dev> pose_stack_block_coord_offset,
      TView<Int, 2, Dev> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, Dev> pose_stack_inter_block_connections,
      TView<Int, 3, Dev> block_type_atom_downstream_of_conn,
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors,
      TView<Real, 3, Dev> block_type_atom_icoors,
      TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors_backup,
      TView<Real, 3, Dev> block_type_atom_icoors_backup,
      TView<Int, 2, Dev> pose_stack_atom_missing) {
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
    if (anc0 == -1 || anc1 == -1 || anc2 == -1
        || pose_stack_atom_missing[pose_ind][anc0]
        || pose_stack_atom_missing[pose_ind][anc1]
        || pose_stack_atom_missing[pose_ind][anc2]) {
      // if any of the ancestors failed to resolve or if any of them
      // are incomplete, then use the backup geometry
      auto get_backup_anc = ([&] TMOL_DEVICE_FUNC(int which_anc) {
        return score::common::resolve_atom_from_uaid<Int, Dev>(
            block_type_atom_ancestors_backup[block_type][atom_ind][which_anc],
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

      if (anc0 == -1 || anc1 == -1 || anc2 == -1
          || pose_stack_atom_missing[pose_ind][anc0]
          || pose_stack_atom_missing[pose_ind][anc1]
          || pose_stack_atom_missing[pose_ind][anc2]) {
        // cannot build an atom if we're missing its ancestors
        return anc_and_geom::empty();
      }
      D = block_type_atom_icoors_backup[block_type][atom_ind][2];      // D
      theta = block_type_atom_icoors_backup[block_type][atom_ind][1];  // theta
      phi = block_type_atom_icoors_backup[block_type][atom_ind][0];    // phi

    } else {
      D = block_type_atom_icoors[block_type][atom_ind][2];      // D
      theta = block_type_atom_icoors[block_type][atom_ind][1];  // theta
      phi = block_type_atom_icoors[block_type][atom_ind][0];    // phi
    }
    return anc_and_geom({anc0, anc1, anc2, D, theta, phi});
  }
};

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
