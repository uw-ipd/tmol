#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/score/common/dispatch.hh>
#include <tmol/score/elec/potentials/dispatch.hh>

#include <tmol/utility/function_dispatch/pybind.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template <tmol::Device Dev, typename Real, typename Int>
void bind_dispatch(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace tmol::utility::function_dispatch;
  using tmol::score::common::NaiveDispatch;

#define ELEC_PYARGS()                                          \
  "x_i"_a, "e_i"_a, "x_j"_a, "e_j"_a, "bonded_path_lengths"_a, \
      "elec_sigmoidal_die_D"_a, "elec_sigmoidal_die_D0"_a,     \
      "elec_sigmoidal_die_S"_a, "elec_min_dis"_a, "elec_max_dis"_a

  add_dispatch_impl<Dev, Real>(
      m,
      "elec",
      &ElecDispatch<common::NaiveDispatch, Dev, Real, Int>::f,
      ELEC_PYARGS());

  add_dispatch_impl<Dev, Real>(
      m,
      "elec_triu",
      &ElecDispatch<common::NaiveTriuDispatch, Dev, Real, Int>::f,
      ELEC_PYARGS());
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
}  // namespace elec
}  // namespace score
}  // namespace tmol
