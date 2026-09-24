#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include "params.hh"

namespace tmol {
namespace score {
namespace metal {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct MetalCoordinationPoseScoreDispatch {
  static auto forward(
      ContextManager& mgr,
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

      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 2, D> conn_atom,
      TView<Int, 1, D> metal_atom,
      TView<Int, 2, D> conn_virt,
      TView<Int, 2, D> conn_key,
      TView<MetalSiteParams<Real>, 2, D> site_params,
      TView<Vec<Int, 2>, 2, D> fan_atoms,
      TView<MetalFanParams<Real>, 2, D> fan_params,
      bool output_block_pair_energies,
      bool compute_derivs)
      -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>>;

  static auto backward(
      ContextManager& mgr,
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

      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 2, D> conn_atom,
      TView<Int, 1, D> metal_atom,
      TView<Int, 2, D> conn_virt,
      TView<Int, 2, D> conn_key,
      TView<MetalSiteParams<Real>, 2, D> site_params,
      TView<Vec<Int, 2>, 2, D> fan_atoms,
      TView<MetalFanParams<Real>, 2, D> fan_params,
      TView<Real, 4, D> dTdV) -> TPack<Vec<Real, 3>, 2, D>;
};

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct MetalCoordinationRotamerScoreDispatch {
  static auto forward(
      ContextManager& mgr,
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

      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 2, D> conn_atom,
      TView<Int, 1, D> metal_atom,
      TView<Int, 2, D> conn_virt,
      TView<Int, 2, D> conn_key,
      TView<MetalSiteParams<Real>, 2, D> site_params,
      TView<Vec<Int, 2>, 2, D> fan_atoms,
      TView<MetalFanParams<Real>, 2, D> fan_params,
      bool output_block_pair_energies,
      bool compute_derivs)
      -> std::tuple<
          TPack<Real, 2, D>,
          TPack<Vec<Real, 3>, 2, D>,
          TPack<Int, 2, D>,
          TPack<Int, 2, D>>;

  static auto backward(
      ContextManager& mgr,
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

      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Int, 2, D> conn_atom,
      TView<Int, 1, D> metal_atom,
      TView<Int, 2, D> conn_virt,
      TView<Int, 2, D> conn_key,
      TView<MetalSiteParams<Real>, 2, D> site_params,
      TView<Vec<Int, 2>, 2, D> fan_atoms,
      TView<MetalFanParams<Real>, 2, D> fan_params,
      TView<Int, 2, D> terms_for_dispatch,  // from forward pass
      TView<Real, 2, D> dTdV) -> TPack<Vec<Real, 3>, 2, D>;
};

}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol
