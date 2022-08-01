
#include <tmol/utility/tensor/pybind.h>

#include <tmol/score/common/complex_dispatch.hh>
#include "test.hh"

namespace tmol {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;

  m.def(
      "exhaustive_dispatch",
      &DispatchTest<
          tmol::score::common::ExhaustiveDispatch,
          Device::CPU,
          double>::f,
      "coords"_a);

  m.def(
      "naive_dispatch",
      &DispatchTest<tmol::score::common::NaiveDispatch, Device::CPU, double>::f,
      "coords"_a);

  m.def(
      "exhaustive_triu_dispatch",
      &DispatchTest<
          tmol::score::common::ExhaustiveTriuDispatch,
          Device::CPU,
          double>::f,
      "coords"_a);

  m.def(
      "naive_triu_dispatch",
      &DispatchTest<
          tmol::score::common::NaiveTriuDispatch,
          Device::CPU,
          double>::f,
      "coords"_a);

  m.def(
      "complex_dispatch",
      &ComplexDispatchTest<
          tmol::score::common::ComplexDispatch,
          Device::CPU,
          int32_t>::f,
      "vals"_a,
      "boundaries"_a);

#ifdef WITH_CUDA

  m.def(
      "exhaustive_dispatch",
      &DispatchTest<
          tmol::score::common::ExhaustiveDispatch,
          Device::CUDA,
          double>::f,
      "coords"_a);

  m.def(
      "exhaustive_triu_dispatch",
      &DispatchTest<
          tmol::score::common::ExhaustiveTriuDispatch,
          Device::CUDA,
          double>::f,
      "coords"_a);

  m.def(
      "naive_dispatch",
      &DispatchTest<tmol::score::common::NaiveDispatch, Device::CUDA, double>::
          f,
      "coords"_a);

  m.def(
      "naive_triu_dispatch",
      &DispatchTest<
          tmol::score::common::NaiveTriuDispatch,
          Device::CUDA,
          double>::f,
      "coords"_a);

#endif
}
}  // namespace tmol
