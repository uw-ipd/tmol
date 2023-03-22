#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/omega/potentials/omega_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template struct OmegaPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct OmegaPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
