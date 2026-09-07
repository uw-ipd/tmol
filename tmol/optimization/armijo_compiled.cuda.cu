#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/optimization/armijo_compiled.impl.hh>

namespace tmol {
namespace optimization {

template struct ArmijoCompiledDispatch<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float>;
template struct ArmijoCompiledDispatch<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double>;

}  // namespace optimization
}  // namespace tmol
