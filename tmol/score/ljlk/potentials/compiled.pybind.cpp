#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "dispatch.hh"
#include "params.pybind.hh"

#include <tmol/utility/function_dispatch/pybind.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;
  using tmol::score::common::NaiveDispatch;

  add_dispatch_impl<Dev, Real>(
      m,
      "lk_isotropic",
      &LKIsotropicDispatch<NaiveDispatch, Dev, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      "type_params"_a,
      "global_params"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "lk_isotropic_triu",
      &LKIsotropicDispatch<NaiveTriuDispatch, Dev, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      "type_params"_a,
      "global_params"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "lj",
      &LJDispatch<NaiveDispatch, Dev, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      "type_params"_a,
      "global_params"_a);

  add_dispatch_impl<Dev, Real>(
      m,
      "lj_triu",
      &LJDispatch<NaiveTriuDispatch, Dev, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      "type_params"_a,
      "global_params"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_dispatch<tmol::Device::CPU, float, int64_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int64_t>(m);

#ifdef WITH_CUDA
  bind_dispatch<tmol::Device::CUDA, float, int64_t>(m);
  bind_dispatch<tmol::Device::CUDA, double, int64_t>(m);
#endif
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
