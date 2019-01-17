#include <pybind11/eigen.h>
#include <torch/torch.h>

#include <tmol/score/common/cubic_hermite_polynomial.hh>

using namespace tmol::score::common;

template <typename Real>
void bind(pybind11::module& m) {
  using namespace pybind11::literals;

  m.def(
      "interpolate",
      &interpolate<Real>,
      "x"_a,
      "x0"_a,
      "p0"_a,
      "dpdx0"_a,
      "x1"_a,
      "p1"_a,
      "dpdx1"_a);
  m.def(
      "interpolate_dt",
      &interpolate_dt<Real>,
      "t"_a,
      "p0"_a,
      "dp0"_a,
      "p1"_a,
      "dp1"_a);
  m.def(
      "interpolate_dx",
      &interpolate_dx<Real>,
      "x"_a,
      "x0"_a,
      "p0"_a,
      "dpdx0"_a,
      "x1"_a,
      "p1"_a,
      "dpdx1"_a);
  m.def(
      "interpolate_t",
      &interpolate_t<Real>,
      "t"_a,
      "p0"_a,
      "dp0"_a,
      "p1"_a,
      "dp1"_a);
  m.def(
      "interpolate_to_zero",
      &interpolate_to_zero<Real>,
      "x"_a,
      "x0"_a,
      "p0"_a,
      "dpdx0"_a,
      "x1"_a);
  m.def(
      "interpolate_to_zero_V_dV",
      &interpolate_to_zero_V_dV<Real>,
      "x"_a,
      "x0"_a,
      "p0"_a,
      "dpdx0"_a,
      "x1"_a);
  m.def(
      "interpolate_to_zero_dt",
      &interpolate_to_zero_dt<Real>,
      "t"_a,
      "p0"_a,
      "dp0"_a);
  m.def(
      "interpolate_to_zero_dx",
      &interpolate_to_zero_dx<Real>,
      "x"_a,
      "x0"_a,
      "p0"_a,
      "dpdx0"_a,
      "x1"_a);
  m.def(
      "interpolate_to_zero_t",
      &interpolate_to_zero_t<Real>,
      "t"_a,
      "p0"_a,
      "dp0"_a);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  // bind<float>(m);
  bind<double>(m);
}
