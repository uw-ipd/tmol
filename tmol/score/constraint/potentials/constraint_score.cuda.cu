#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/constraint/potentials/constraint_score.impl.hh>

namespace tmol {
namespace score {
namespace constraint {
namespace potentials {

template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float>;
template struct GetTorsionAngleDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double>;

}  // namespace potentials
}  // namespace constraint
}  // namespace score
}  // namespace tmol
