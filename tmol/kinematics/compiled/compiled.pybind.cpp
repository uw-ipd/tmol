#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "compiled.hh"

#include <tmol/utility/function_dispatch/pybind.hh>

namespace tmol {
namespace kinematics {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "forward_kin",
      &ForwardKinDispatch<Dev, Real, Int>::f,
      "dofs"_a,
      "doftypes"_a,
      "nodes"_a,
      "scans"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "dof_transforms",
      &DOFTransformsDispatch<Dev, Real, Int>::f,
      "dofs"_a,
      "doftypes"_a);

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
