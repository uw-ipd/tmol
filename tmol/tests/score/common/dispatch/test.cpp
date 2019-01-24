#include <tmol/score/common/dispatch.cpu.impl.hh>
#include <tmol/score/common/dispatch.hh>

#include "test.impl.hh"

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
