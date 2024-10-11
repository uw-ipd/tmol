#pragma once

#include <Eigen/Core>
#include <tuple>

#include <torch/torch.h>

#include <pybind11/eigen.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/pybind.h>
#include <tmol/utility/function_dispatch/pybind.hh>

#include "common.hh"
#include "params.hh"

#define HomogeneousTransform Eigen::Matrix<Real, 4, 4>
#define KintreeDof Eigen::Matrix<Real, 9, 1>
#define Coord Eigen::Matrix<Real, 3, 1>

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// dofs -> xyz + HTs
//   involves a segmented scan over HTs
template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree)
      -> std::tuple<TPack<Coord, 1, D>, TPack<HomogeneousTransform, 1, D> >;
};

// xyz -> dofs
template <tmol::Device D, typename Real, typename Int>
struct InverseKinDispatch {
  static auto f(
      TView<Coord, 1, D> coord,
      TView<Int, 1, D> parent,
      TView<Int, 1, D> frame_x,
      TView<Int, 1, D> frame_y,
      TView<Int, 1, D> frame_z,
      TView<Int, 1, D> doftype) -> TPack<KintreeDof, 1, D>;
};

// dEdx -> dEddof
//   involves a segmented scan over f1f2s
template <tmol::Device D, typename Real, typename Int>
struct KinDerivDispatch {
  static auto f(
      TView<Coord, 1, D> dVdx,
      TView<HomogeneousTransform, 1, D> hts,
      TView<KintreeDof, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinForestGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinForestParams<Int>, 1, D> kintree) -> TPack<KintreeDof, 1, D>;
};

//
//
// template <template <tmol::Device> class DeviceOps, tmol::Device D, typename
// Int> struct FixJumpNodes {
//   static void f(
//       TView<Int, 1, D> parents,
//       TView<Int, 1, D> frame_x,
//       TView<Int, 1, D> frame_y,
//       TView<Int, 1, D> frame_z,
//       TView<Int, 1, D> roots,
//       TView<Int, 1, D> jumps);
// };

#undef HomogeneousTransform
#undef KintreeDof
#undef Coord

}  // namespace kinematics
}  // namespace tmol
