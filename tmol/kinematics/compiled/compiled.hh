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

template <tmol::Device D, typename Real, typename Int>
struct BackwardKinDispatch {
  static auto f(
      TView<Vec<Real, 3>, 1, D> ht,
      TView<Int, 1, D> doftypes,
      TView<Int, 1, D> parents,
      TView<Int, 1, D> frame_x,
      TView<Int, 1, D> frame_y,
      TView<Int, 1, D> frame_z,
      TView<Vec<Real, 9>, 1, D> dofs) -> TPack<HomogeneousTransform, 1, D>;
};

template <tmol::Device D, typename Real, typename Int>
struct f1f2ToDerivsDispatch {
  static auto f(
      TView<HomogeneousTransform, 1, D> hts,
      TView<Vec<Real, 9>, 1, D> dofs,
      TView<Int, 1, D> doftypes,
      TView<Int, 1, D> parents,
      TView<Vec<Real, 6>, 1, D> f1f2s) -> TPack<Vec<Real, 9>, 1, D>;
};

#undef HomogeneousTransform

}  // namespace kinematics
}  // namespace tmol
