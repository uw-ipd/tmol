#include <pybind11/pybind11.h>

#include <tmol/utility/tensor/pybind.h>

#include "hybrid.hh"

namespace tmol {
namespace tests {
namespace utility {
namespace cpp_extension {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("sum", &sum_tensor<float, tmol::Device::CPU>::f, "t"_a);
#ifdef WITH_CUDA
  m.def("sum", &sum_tensor<float, tmol::Device::CUDA>::f, "t"_a);
#endif
}
}
}
}
}