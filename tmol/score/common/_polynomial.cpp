#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <tmol/score/common/polynomial.hh>

namespace tmol {
namespace score {
namespace common {

template <int POrd, typename Real>
void bind_polynomial(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def("poly_v", &poly_v<POrd, Real>, "x"_a, "coeffs"_a);
  m.def("poly_v_d", &poly_v_d<POrd, Real>, "x"_a, "coeffs"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_polynomial<2, float>(m);
  bind_polynomial<3, float>(m);
  bind_polynomial<4, float>(m);
  bind_polynomial<5, float>(m);
  bind_polynomial<6, float>(m);
  bind_polynomial<7, float>(m);
  bind_polynomial<8, float>(m);
  bind_polynomial<9, float>(m);
  bind_polynomial<10, float>(m);
  bind_polynomial<11, float>(m);

  bind_polynomial<2, double>(m);
  bind_polynomial<3, double>(m);
  bind_polynomial<4, double>(m);
  bind_polynomial<5, double>(m);
  bind_polynomial<6, double>(m);
  bind_polynomial<7, double>(m);
  bind_polynomial<8, double>(m);
  bind_polynomial<9, double>(m);
  bind_polynomial<10, double>(m);
  bind_polynomial<11, double>(m);
}

}  // namespace common
}  // namespace score
}  // namespace tmol
