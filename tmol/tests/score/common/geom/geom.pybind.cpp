#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <tmol/score/common/geom.hh>

using namespace tmol::score::common;

template <typename Real>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def("distance_V", &distance<Real>::V, "A"_a, "B"_a);
  m.def(
      "distance_V_dV",
      [](Vec<Real, 3> A, Vec<Real, 3> B) {
        return distance<Real>::V_dV(A, B).astuple();
      },
      "A"_a,
      "B"_a);

  m.def("interior_angle_V", &interior_angle<Real>::V, "A"_a, "B"_a);
  m.def(
      "interior_angle_V_dV",
      [](Vec<Real, 3> A, Vec<Real, 3> B) {
        return interior_angle<Real>::V_dV(A, B).astuple();
      },
      "A"_a,
      "B"_a);

  m.def("cos_interior_angle_V", &cos_interior_angle<Real>::V, "A"_a, "B"_a);
  m.def(
      "cos_interior_angle_V_dV",
      [](Vec<Real, 3> A, Vec<Real, 3> B) {
        return cos_interior_angle<Real>::V_dV(A, B).astuple();
      },
      "A"_a,
      "B"_a);

  m.def(
      "dihedral_angle_V", &dihedral_angle<Real>::V, "I"_a, "J"_a, "K"_a, "L"_a);
  m.def(
      "dihedral_angle_V_dV",
      [](Vec<Real, 3> I, Vec<Real, 3> J, Vec<Real, 3> K, Vec<Real, 3> L) {
        return dihedral_angle<Real>::V_dV(I, J, K, L).astuple();
      },
      "I"_a,
      "J"_a,
      "K"_a,
      "L"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  bind<double>(m);
}
