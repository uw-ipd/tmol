#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/constraint/potentials/constraint_score.impl.hh>

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    float>;
template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    double>;

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
