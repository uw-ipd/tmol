#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "compiled.hh"

#include <tmol/utility/function_dispatch/pybind.hh>

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "cartbonded_length",
      &CartBondedLengthDispatch<Dev, Real, Int>::f,
      "coords"_a,
      "atom_pair_indices"_a,
      "parameter_indices"_a,
      "K"_a,
      "x0"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "cartbonded_angle",
      &CartBondedAngleDispatch<Dev, Real, Int>::f,
      "coords"_a,
      "atom_triple_indices"_a,
      "parameter_indices"_a,
      "K"_a,
      "x0"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "cartbonded_torsion",
      &CartBondedTorsionDispatch<Dev, Real, Int>::f,
      "coords"_a,
      "atom_quad_indices"_a,
      "parameter_indices"_a,
      "K"_a,
      "x0"_a,
      "period"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "cartbonded_hxltorsion",
      &CartBondedHxlTorsionDispatch<Dev, Real, Int>::f,
      "coords"_a,
      "atom_quad_indices"_a,
      "parameter_indices"_a,
      "k1"_a,
      "k2"_a,
      "k3"_a,
      "phi1"_a,
      "phi2"_a,
      "phi3"_a);
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
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
