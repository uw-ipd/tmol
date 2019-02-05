#include <pybind11/eigen.h>
#include <tmol/utility/tensor/pybind.h>
#include <torch/torch.h>

#include <tmol/score/ljlk/potentials/lk_ball.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real>
void bind_potentials(pybind11::module& m) {
  using namespace pybind11::literals;
  using namespace pybind11;

  m.def("build_don_water_V", &build_don_water<Real>::V, "D"_a, "H"_a, "dist"_a);
  m.def(
      "build_don_water_dV", &build_don_water<Real>::dV, "D"_a, "H"_a, "dist"_a);

  m.def(
      "build_acc_water_V",
      &build_acc_water<Real>::V,
      "A"_a,
      "B"_a,
      "B0"_a,
      "dist"_a,
      "angle"_a,
      "torsion"_a);

  m.def(
      "build_acc_water_dV",
      &build_acc_water<Real>::dV,
      "A"_a,
      "B"_a,
      "B0"_a,
      "dist"_a,
      "angle"_a,
      "torsion"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_potentials<double>(m); }

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
