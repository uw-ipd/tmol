#pragma once

#include <Eigen/Core>
#include <tuple>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include "common.hh"

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define Dofs Eigen::Matrix<Real, 9, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<Dofs, 1, D> dofs,
      TView<Int, 1, D> doftypes,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<Vec<Int, 2>, 1, tmol::Device::CPU> gens)
      -> TPack<HomogeneousTransform, 1, D>;
};

template <tmol::Device D, typename Real, typename Int>
struct DOFTransformsDispatch {
  static auto f(TView<Vec<Real, 9>, 1, D> dofs, TView<Int, 1, D> doftypes)
      -> TPack<HomogeneousTransform, 1, D>;
};

template <tmol::Device D, typename Real, typename Int>
struct BackwardKinDispatch {
  static auto f(
      TView<Coord, 1, D> coord,
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
      TView<Dofs, 1, D> dofs,
      TView<Int, 1, D> doftypes,
      TView<Int, 1, D> parents,
      TView<Vec<Real, 6>, 1, D> f1f2s) -> TPack<Vec<Real, 9>, 1, D>;
};

template <tmol::Device D, typename Real, typename Int>
struct SegscanF1f2sDispatch {
  static auto f(
      TView<Vec<Real, 6>, 1, D> f1f2s,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<Vec<Int, 2>, 1, tmol::Device::CPU> gens) -> void;
};

#undef HomogeneousTransform
#undef Dofs
#undef Coord

}  // namespace kinematics
}  // namespace tmol
