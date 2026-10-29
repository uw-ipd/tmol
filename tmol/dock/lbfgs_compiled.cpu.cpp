#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/dock/lbfgs_compiled.impl.hh>

namespace tmol {
namespace dock {

template struct LBFGSDispatch<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float>;
template struct LBFGSDispatch<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double>;

}  // namespace dock
}  // namespace tmol
