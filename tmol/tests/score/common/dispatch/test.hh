#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/tuple.hh>
#include <tuple>

namespace tmol {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <template <Device> class Dispatch, Device D, typename Real>
struct DispatchTest {
  static auto f(TView<Vec<Real, 3>, 1, D> coords)
      -> std::tuple<at::Tensor, at::Tensor>;
};
}  // namespace tmol
