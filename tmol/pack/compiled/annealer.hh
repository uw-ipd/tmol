#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

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
      int const chunk_size,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      TView<Int, 1, D> pose_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<int32_t, 1, D> block_ind_for_rot,
      TView<int32_t, 2, D> sparse_inds,
      TView<Real, 1, D> sparse_energies)
      -> std::tuple<
          TPack<Real, 1, D>,
          TPack<int64_t, 3, D>,
          TPack<int64_t, 1, D>,
          TPack<Real, 1, D> >;
};

template <tmol::Device D>
struct AnnealerDispatch {
  static auto forward(
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
