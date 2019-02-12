#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "../../bonded_atom.pybind.hh"
#include "water.pybind.hh"

#include "water.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "attached_waters_forward",
      attached_waters<Real, Int, tmol::Device::CPU, 4>::forward,
      "coords"_a,
      "indexed_bonds"_a,
      "atom_types"_a,
      "global_params"_a);

  m.def(
      "attached_waters_backward",
      attached_waters<Real, Int, tmol::Device::CPU, 4>::backward,
      "dE_dW"_a,
      "coords"_a,
      "indexed_bonds"_a,
      "atom_types"_a,
      "global_params"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_dispatch<float, int64_t>(m);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
