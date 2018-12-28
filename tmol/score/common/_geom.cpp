#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <tmol/score/common/geom.hh>

using namespace tmol::score::common;

template <typename Real>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def("interior_angle_V", &interior_angle_V<Real>, "v1"_a, "v2"_a);

  m.def("interior_angle_V_dV", &interior_angle_V_dV<Real>, "v1"_a, "v2"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind<float>(m);
  bind<double>(m);
}
