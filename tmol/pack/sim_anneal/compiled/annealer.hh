#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
//#include <tmol/pack/compiled/params.hh>

namespace tmol {
namespace pack {
namespace sim_anneal {
namespace compiled {

template <
    template <tmol::Device>
    class Dispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct PickRotamers {
  static auto f(
      TView<Real, 4, D> context_coords,
      TView<Int, 2, D> context_block_type,
      TView<Int, 1, D> pose_id_for_context,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Real, 3, D> rotamer_coords)
      -> std::tuple<TPack<Real, 3, D>, TPack<Int, 2, D>, TPack<Int, 1, D> >;
};

}  // namespace compiled
}  // namespace sim_anneal
}  // namespace pack
}  // namespace tmol
