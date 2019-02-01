#include <pybind11/pybind11.h>

#include <tmol/utility/tensor/pybind.h>

#include "hybrid.hh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("sumx", &sumx<float, tmol::Device::CPU>::f, "t"_a);
#ifdef WITH_CUDA
  m.def("sumx", &sumx<float, tmol::Device::CUDA>::f, "t"_a);
#endif
}
