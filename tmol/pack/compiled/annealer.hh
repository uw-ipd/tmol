#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct InteractionGraphBuilder {
  static auto f(
      ContextManager& mgr,
      int const bump_check,
      int const chunk_size,
      int const max_n_block_types,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      TView<Int, 1, D> pose_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<int32_t, 1, D> block_ind_for_rot,
      TView<int32_t, 2, D> sparse_inds,
      TView<Real, 1, D> sparse_energies,
      int const verbose)
      -> std::tuple<
          TPack<
              int64_t,
              1,
              tmol::Device::CPU>,  // max_n_bump_checked_rotamers_per_pose
          TPack<Int, 1, D>,        // n_molten_blocks_per_pose
          TPack<Int, 1, D>,        // n_bc_rots_per_pose
          TPack<Int, 1, D>,        // bc_rot_offset_for_pose
          TPack<Int, 2, D>,        // n_bc_rots_for_molten_block
          TPack<Int, 2, D>,        // bc_rot_offset_for_molten_block
          TPack<Int, 1, D>,        // molten_block_ind_for_bc_rot
          TPack<int64_t, 2, D>,    // rotamer_for_nonmolten_block
          TPack<int64_t, 1, D>,    // bc_rot_to_orig_rot

          TPack<Real, 1, D>,  // bg/bg energies
          TPack<Real, 1, D>,  // energy1b
          TPack<int64_t, 3, D>,
          TPack<int64_t, 1, D>,
          TPack<Real, 1, D> >;  // energy2b
};

template <tmol::Device D>
struct AnnealerDispatch {
  static auto forward(
      ContextManager& mgr,
      int max_n_rotamers_per_pose,
      TView<int, 1, D> pose_n_res,
      TView<int, 1, D> pose_n_rotamers,
      TView<int, 1, D> pose_rotamer_offset,
      TView<int, 2, D> n_rotamers_for_res,
      TView<int, 2, D> oneb_offsets,
      TView<int, 1, D> res_for_rot,
      int32_t chunk_size,
      TView<int64_t, 3, D> chunk_offset_offsets,
      TView<int64_t, 1, D> chunk_offsets,
      TView<float, 1, D> energy1b,
      TView<float, 1, D> energy2b)
      -> std::tuple<TPack<float, 2, D>, TPack<int, 3, D> >;
};

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
