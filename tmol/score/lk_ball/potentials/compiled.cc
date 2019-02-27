#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/score/common/dispatch.cpu.impl.hh>
#include <tmol/utility/function_dispatch/pybind.hh>

#include <tmol/score/ljlk/potentials/params.pybind.hh>
#include "../../bonded_atom.pybind.hh"
#include "datatypes.pybind.hh"

#include "dispatch.impl.hh"
#include "water.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "attached_waters_forward",
      attached_waters<Real, Int, Dev, 4>::forward,
      "coords"_a,
      "indexed_bonds"_a,
      "atom_types"_a,
      "global_params"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "attached_waters_backward",
      attached_waters<Real, Int, Dev, 4>::backward,
      "dE_dW"_a,
      "coords"_a,
      "indexed_bonds"_a,
      "atom_types"_a,
      "global_params"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "lk_ball_V",
      &LKBallDispatch<common::NaiveDispatch, Dev, Real, Int>::V,
      "coords_i"_a,
      "coords_j"_a,
      "waters_i"_a,
      "waters_j"_a,
      "atom_type_i"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      "type_params"_a,
      "global_lkb_params"_a,
      "global_lj_params"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "lk_ball_dV",
      &LKBallDispatch<common::NaiveDispatch, Dev, Real, Int>::dV,
      "coords_i"_a,
      "coords_j"_a,
      "waters_i"_a,
      "waters_j"_a,
      "atom_type_i"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      "type_params"_a,
      "global_lkb_params"_a,
      "global_lj_params"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind_dispatch<tmol::Device::CPU, float, int64_t>(m);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
