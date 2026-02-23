#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "params.hh"

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct DisulfidePoseScoreDispatch {
  static auto forward(
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

      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<bool, 2, D> disulfide_conns,
      TView<Int, 3, D> block_type_atom_downstream_of_conn,

      TView<DisulfideGlobalParams<Real>, 1, D> global_params,
      bool output_block_pair_energies,
      bool compute_derivs)
      -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>>;

  static auto backward(
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

      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<bool, 2, D> disulfide_conns,
      TView<Int, 3, D> block_type_atom_downstream_of_conn,

      TView<DisulfideGlobalParams<Real>, 1, D> global_params,

      TView<Real, 4, D> dTdV  // n_terms x n_dispatch_total
      ) -> TPack<Vec<Real, 3>, 2, D>;
};

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct DisulfideRotamerScoreDispatch {
  static auto forward(
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

      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<bool, 2, D> disulfide_conns,
      TView<Int, 3, D> block_type_atom_downstream_of_conn,

      TView<DisulfideGlobalParams<Real>, 1, D> global_params,
      bool output_block_pair_energies,
      bool compute_derivs)
      -> std::tuple<
          TPack<Real, 2, D>,
          TPack<Vec<Real, 3>, 2, D>,
          TPack<Int, 2, D>,
          TPack<Int, 2, D>>;

  static auto backward(
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

      TView<Int, 2, D> pose_stack_block_type,
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<bool, 2, D> disulfide_conns,
      TView<Int, 3, D> block_type_atom_downstream_of_conn,

      TView<DisulfideGlobalParams<Real>, 1, D> global_params,
      TView<Int, 2, D> dispatch_indices,            // from forward pass
      TView<Int, 2, D> conns_for_dispatch_indices,  // from forward pass

      TView<Real, 2, D> dTdV  // n_terms x n_dispatch_total
      ) -> TPack<Vec<Real, 3>, 2, D>;
};

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
