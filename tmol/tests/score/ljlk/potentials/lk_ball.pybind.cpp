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

  m.def("build_don_water", &build_don_water<Real>::V, "d"_a, "h"_a, "dist"_a);

  m.def(
      "build_acc_waters",
      &build_acc_waters<Real, 2>::V,
      "a"_a,
      "b"_a,
      "b0"_a,
      "dist"_a,
      "angle"_a,
      "tors"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind_potentials<double>(m); }

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol