// #include <tmol/pack/rotamer/dispatch.hh>
#include <tmol/pack/rotamer/dispatch.impl.hh>
#include <tmol/score/common/complex_dispatch.cuda.impl.cuh>
#include "test.impl.hh"

namespace tmol {

template struct DunbrackChiSamplerTester<
    tmol::score::common::ComplexDispatch,
    Device::CUDA,
    float,
    int32_t>;

}
