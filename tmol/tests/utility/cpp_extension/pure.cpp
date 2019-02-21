#include <pybind11/pybind11.h>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/tensor/pybind.h>

#include <cppitertools/range.hpp>

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

using tmol::TView;

template <typename Real, tmol::Device D>
struct sum {};

template <typename Real>
struct sum<Real, tmol::Device::CPU> {
  static const tmol::Device D = tmol::Device::CPU;

  static auto f(TView<Real, 1, D> t) -> Real {
    Real v = 0;
    using iter::range;

    for (auto i : range(t.size(0))) {
      v += t[i];
    }

    return v;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("sum", &sum<float, tmol::Device::CPU>::f, "t"_a);
}
}
}
}
}