#include <pybind11/pybind11.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

#include <cppitertools/range.hpp>

using tmol::TView;

template <typename Real>
auto sum(TView<Real, 1> t) -> Real {
  Real v = 0;
  using iter::range;

  for (auto i : range(t.size(0))) {
    v += t[i];
  }

  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("sum", &sum<float>, "t"_a);
}
