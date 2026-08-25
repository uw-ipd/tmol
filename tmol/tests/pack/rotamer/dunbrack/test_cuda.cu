#include <tmol/pack/rotamer/dunbrack/dispatch.impl.hh>
#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include "test.impl.hh"

namespace tmol {

template struct DunbrackChiSamplerTester<
    tmol::score::common::DeviceOperations,
    Device::CUDA,
    float,
    int32_t>;

}
