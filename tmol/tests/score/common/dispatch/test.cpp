#include <tmol/score/common/dispatch.cpu.impl.hh>
#include <tmol/score/common/dispatch.hh>
#include <tmol/score/common/complex_dispatch.cpu.impl.hh>

#include "test.impl.hh"

namespace tmol {
template struct DispatchTest<
    tmol::score::common::ExhaustiveDispatch,
    Device::CPU,
    double>;
template struct DispatchTest<
    tmol::score::common::NaiveDispatch,
    Device::CPU,
    double>;
template struct DispatchTest<
    tmol::score::common::ExhaustiveTriuDispatch,
    Device::CPU,
    double>;
template struct DispatchTest<
    tmol::score::common::NaiveTriuDispatch,
    Device::CPU,
    double>;
template struct ComplexDispatchTest<
  tmol::score::common::ComplexDispatch,
  Device::CPU,
  int32_t>;
}  // namespace tmol
