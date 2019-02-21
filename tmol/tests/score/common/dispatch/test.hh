#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/dispatch.hh>
#include <tuple>

namespace tmol {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <template <Device> class Dispatch, Device D, typename Real>
struct DispatchTest {
  static auto f(TView<Vec<Real, 3>, 1, D> coords)
      -> std::tuple<TPack<int64_t, 2, D>, TPack<float, 1, D>>;
};
}  // namespace tmol
