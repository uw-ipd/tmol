#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "compiled.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "cartbonded_length",
      &CartBondedLengthDispatch<Dev, Real, Int>::f,
      "atom_pair_indices"_a, "parameter_indices"_a, "coords"_a, "K"_a, "x0"_a );

  m.def(
      "cartbonded_angle",
      &CartBondedAngleDispatch<Dev, Real, Int>::f,
      "atom_triple_indices"_a, "parameter_indices"_a, "coords"_a, "K"_a, "x0"_a );

  m.def(
      "cartbonded_torsion",
      &CartBondedTorsionDispatch<Dev, Real, Int>::f,
      "atom_quad_indices"_a, "parameter_indices"_a, "coords"_a, "K"_a, "x0"_a , "period"_a );

  m.def(
      "cartbonded_hxltorsion",
      &CartBondedHxlTorsionDispatch<Dev, Real, Int>::f,
      "atom_quad_indices"_a, "parameter_indices"_a, "coords"_a,
	  "k1"_a, "k2"_a, "k3"_a,
	  "phi1"_a, "phi2"_a, "phi3"_a );
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_dispatch<tmol::Device::CPU, float, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, float, int64_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int64_t>(m);

#ifdef WITH_CUDA
  bind_dispatch<tmol::Device::CUDA, float, int32_t>(m);
  bind_dispatch<tmol::Device::CUDA, double, int32_t>(m);
#endif
}


}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
