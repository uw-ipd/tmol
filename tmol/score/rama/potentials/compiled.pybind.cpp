#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/extension.h>

#include <tmol/score/common/forall_dispatch.hh>
#include "dispatch.hh"

#include <tmol/utility/function_dispatch/pybind.hh>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "rama",
      &RamaDispatch<common::ForallDispatch, Dev, Real, Int>::f,
      "coords"_a,
      "phi_indices"_a,
      "psi_indices"_a,
      "table_indices"_a,
      "tables"_a, 
      "bbstarts"_a,
      "bbsteps"_a );
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


}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
