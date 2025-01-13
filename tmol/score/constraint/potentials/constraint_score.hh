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

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device>
    class DeviceOps,
    tmol::Device D,
    typename Real>
struct GetTorsionAngleDispatch {
  static auto forward(TView<Vec<Real, 3>, 2, D> coords)
      -> std::tuple<TPack<Real, 1, D>, TPack<Vec<Real, 3>, 2, D>>;
};

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
