#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include "dispatch.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <tmol::Device D, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using tmol::score::common::NaiveDispatch;

  m.def(
      "lk_isotropic",
      &LKIsotropicDispatch<NaiveDispatch, D, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LKTypeParams_pyargs(),
      LJGlobalParams_pyargs());

  m.def(
      "lk_isotropic_triu",
      &LKIsotropicDispatch<NaiveTriuDispatch, D, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LKTypeParams_pyargs(),
      LJGlobalParams_pyargs());

  m.def(
      "lj",
      &LJDispatch<NaiveDispatch, D, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LJTypeParams_pyargs(),
      LJGlobalParams_pyargs());

  m.def(
      "lj_triu",
      &LJDispatch<NaiveTriuDispatch, D, Real, Int>::f,
      "coords_i"_a,
      "atom_type_i"_a,
      "coords_j"_a,
      "atom_type_j"_a,
      "bonded_path_lengths"_a,
      LJTypeParams_pyargs(),
      LJGlobalParams_pyargs());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_dispatch<tmol::Device::CPU, float, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, float, int64_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int32_t>(m);
  bind_dispatch<tmol::Device::CPU, double, int64_t>(m);

#ifdef WITH_CUDA
  bind_dispatch<tmol::Device::CUDA, float, int32_t>(m);
  bind_dispatch<tmol::Device::CUDA, float, int64_t>(m);
  bind_dispatch<tmol::Device::CUDA, double, int32_t>(m);
  bind_dispatch<tmol::Device::CUDA, double, int64_t>(m);
#endif
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
