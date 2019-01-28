
#include <tmol/utility/tensor/pybind.h>

#include "test.hh"

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
      "exhaustive_omp_dispatch",
      &DispatchTest<
          tmol::score::common::ExhaustiveOMPDispatch,
          Device::CPU,
          double>::f,
      "coords"_a);

  m.def(
      "aabb_dispatch",
      &DispatchTest<
          tmol::score::common::AABBDispatch,
          Device::CPU,
          double>::f,
      "coords"_a);


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
