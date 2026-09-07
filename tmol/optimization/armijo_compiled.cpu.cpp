#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/optimization/armijo_compiled.impl.hh>

namespace tmol {
namespace optimization {

template struct ArmijoCompiledDispatch<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float>;
template struct ArmijoCompiledDispatch<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double>;

}  // namespace optimization
}  // namespace tmol
