#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <tmol/score/common/geom.hh>

using namespace tmol::score::common;

template <typename Real>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def("distance_V", &distance_V<Real>, "A"_a, "B"_a);
  m.def("distance_V_dV", &distance_V_dV<Real>, "A"_a, "B"_a);

  m.def("interior_angle_V", &interior_angle_V<Real>, "A"_a, "B"_a);
  m.def("interior_angle_V_dV", &interior_angle_V_dV<Real>, "A"_a, "B"_a);

  m.def("cos_interior_angle_V", &cos_interior_angle_V<Real>, "A"_a, "B"_a);
  m.def(
      "cos_interior_angle_V_dV", &cos_interior_angle_V_dV<Real>, "A"_a, "B"_a);

  m.def(
      "dihedral_angle_V", &dihedral_angle_V<Real>, "I"_a, "J"_a, "K"_a, "L"_a);
  m.def(
      "dihedral_angle_V_dV",
      &dihedral_angle_V_dV<Real>,
      "I"_a,
      "J"_a,
      "K"_a,
      "L"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind<double>(m);
}
