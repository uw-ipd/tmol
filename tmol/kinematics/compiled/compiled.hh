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

namespace tmol {
namespace kinematics {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define HT Eigen::Matrix<Real, 4, 4>
#define Coord Eigen::Matrix<Real, 3, 1>

// dofs -> xyz + HTs
//   involves a segmented scan over HTs
template <tmol::Device D, typename Real, typename Int>
struct ForwardKinDispatch {
  static auto f(
      TView<DofTypes<Real>, 1, D> dofs,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinTreeGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinTreeParams<Int>, 1, D> kintree)
      -> std::tuple<TPack<Coord, 1, D>, TPack<HT, 1, D> >;
};

// xyz -> dofs
template <tmol::Device D, typename Real, typename Int>
struct InverseKinDispatch {
  static auto f(
      TView<Coord, 1, D> coord, TView<KinTreeParams<Int>, 1, D> kintree)
      -> TPack<DofTypes<Real>, 1, D>;
};

// dEdx -> dEddof
//   involves a segmented scan over f1f2s
template <tmol::Device D, typename Real, typename Int>
struct KinDerivDispatch {
  static auto f(
      TView<Coord, 1, D> dVdx,
      TView<HT, 1, tmol::Device::CPU> hts,
      TView<Int, 1, D> nodes,
      TView<Int, 1, D> scans,
      TView<KinTreeGenData<Int>, 1, tmol::Device::CPU> gens,
      TView<KinTreeParams<Int>, 1, D> kintree) -> TPack<DofTypes<Real>, 1, D>;
};

// pybind-ings for inverse kinematics
// - not part of the evaluation graph but is used in setup
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "inverse_kin",
      &InverseKinDispatch<Dev, Real, Int>::f,
      "coords"_a,
      "parents"_a,
      "doftypes"_a,
      "frame_x"_a,
      "frame_y"_a,
      "frame_z"_a,
      "dofs"_a);
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_dispatch<tmol::Device::CPU, float, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int32_t>(m);

#ifdef WITH_CUDA
  bind_dispatch<tmol::Device::CUDA, float, int32_t>(m);
  bind_dispatch<tmol::Device::CUDA, double, int32_t>(m);
#endif
}

#undef HomogeneousTransform
#undef Dofs
#undef Coord

}  // namespace kinematics
}  // namespace tmol
