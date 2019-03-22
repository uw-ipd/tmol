#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCollection.h>
#include <tmol/utility/tensor/TensorPack.h>

#include "common.hh"

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<Vec<Real, 9>, 1, D> dofs,
      TView<Int, 1, D> doftypes,
      TCollection<Int, 1, D> nodes,
      TCollection<Int, 1, D> scans) -> TPack<HomogeneousTransform, 1, D>;
};

template <tmol::Device D, typename Real, typename Int>
struct DOFTransformsDispatch {
  static auto f(TView<Vec<Real, 9>, 1, D> dofs, TView<Int, 1, D> doftypes)
      -> TPack<HomogeneousTransform, 1, D>;
};

#undef HomogeneousTransform

}  // namespace kinematics
}  // namespace tmol
