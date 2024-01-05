#pragma once
#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/io/details/compiled/his_taut_params.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device Dev,
    typename Real,
    typename Int>
struct ResolveHisTaut {
  static auto f(
      TView<Vec<Real, 3>, 3, Dev> coords,
      TView<Int, 2, Dev> res_types,
      TView<Int, 2, Dev> res_type_variants,
      TView<int64_t, 1, Dev> his_pose_ind,
      TView<int64_t, 1, Dev> his_res_ind,
      TView<Int, 3, Dev> atom_is_present,
      TView<HisAtomIndsInCanonicalOrdering<Int>, 1, Dev> his_atom_inds,
      TView<int64_t, 3, Dev> his_remapping_dst_index) -> TPack<int64_t, 2, Dev>;
};

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
