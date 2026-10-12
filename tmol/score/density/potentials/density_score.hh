#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace score {
namespace density {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real>
struct DensityScoreDispatch {
  // Energy -weight * S(x) of each listed atom, summed into its group, and the
  // gradient of each listed atom's energy.
  static auto forward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, D> coords,
      TView<Real, 3, D> coeffs,
      TView<Real, 1, D> origin,
      TView<Real, 2, D> inv_basis,
      TView<Real, 1, D> pad,
      bool periodic,
      TView<int64_t, 1, D> atom_index,
      TView<int64_t, 1, D> atom_group,
      TView<Real, 1, D> atom_weight,
      int64_t n_groups)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 1, D>>;

  // dE/dcoords: each listed atom's gradient scaled by its group's incoming
  // gradient. Listed atoms are distinct.
  static auto backward(
      ContextManager& mgr,
      TView<Vec<Real, 3>, 1, D> dE_dx,
      TView<int64_t, 1, D> atom_index,
      TView<int64_t, 1, D> atom_group,
      TView<Real, 1, D> group_grad,
      int64_t n_coords) -> TPack<Vec<Real, 3>, 1, D>;
};

}  // namespace potentials
}  // namespace density
}  // namespace score
}  // namespace tmol
