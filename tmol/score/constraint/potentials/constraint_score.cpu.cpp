#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/constraint/potentials/constraint_score.impl.hh>

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    float>;
template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    double>;

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
