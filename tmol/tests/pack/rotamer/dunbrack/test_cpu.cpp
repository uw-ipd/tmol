
#include <tmol/pack/rotamer/dunbrack/dispatch.impl.hh>
#include <tmol/score/common/device_operations.cpu.impl.hh>
#include "test.impl.hh"

namespace tmol {

template struct DunbrackChiSamplerTester<
    tmol::score::common::DeviceOperations,
    Device::CPU,
    float,
    int32_t>;

}
