#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/metal/potentials/metal_coordination_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace metal {
namespace potentials {

template struct MetalCoordinationPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct MetalCoordinationPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

template struct MetalCoordinationRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct MetalCoordinationRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;
}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol
