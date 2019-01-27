#include <tmol/score/common/dispatch.cuda.impl.cuh>
#include <tmol/score/common/dispatch.hh>

#include "test.impl.hh"

namespace tmol {
template struct DispatchTest<
    tmol::score::common::ExhaustiveDispatch,
    Device::CUDA,
    double>;

template struct DispatchTest<
    tmol::score::common::ExhaustiveTriuDispatch,
    Device::CUDA,
    double>;

template struct DispatchTest<
    tmol::score::common::NaiveDispatch,
    Device::CUDA,
    double>;

template struct DispatchTest<
    tmol::score::common::NaiveTriuDispatch,
    Device::CUDA,
    double>;
}  // namespace tmol
