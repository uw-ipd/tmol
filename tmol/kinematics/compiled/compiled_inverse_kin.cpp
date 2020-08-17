#pragma once

#include <Eigen/Core>
#include <tuple>

#include <torch/torch.h>

#include <pybind11/eigen.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/pybind.h>
#include <tmol/utility/function_dispatch/pybind.hh>

#include "common_dispatch.hh"
namespace tmol {
namespace kinematics {

// pybind-ings for inverse kinematics
// - not part of the evaluation graph but is used in setup
template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "inverse_kin",
      &InverseKinDispatch<Dev, Real, Int>::f,
      "coords"_a,
      "parent"_a,
      "frame_x"_a,
      "frame_y"_a,
      "frame_z"_a,
      "doftype"_a);
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

}  // namespace kinematics
}  // namespace tmol
