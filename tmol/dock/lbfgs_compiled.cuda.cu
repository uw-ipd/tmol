#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/dock/lbfgs_compiled.impl.hh>

namespace tmol {
namespace dock {

template struct LBFGSDispatch<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float>;
template struct LBFGSDispatch<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double>;

}  // namespace dock
}  // namespace tmol
