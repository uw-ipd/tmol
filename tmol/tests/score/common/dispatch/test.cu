#include <tmol/score/common/dispatch.cuda.impl.cuh>
#include <tmol/score/common/dispatch.hh>

#include "test.impl.hh"

template struct DispatchTest<
    tmol::score::common::ExhaustiveDispatch,
    Device::CUDA,
    double>;

template struct DispatchTest<
    tmol::score::common::ExhaustiveTriuDispatch,
    Device::CUDA,
    double>;
