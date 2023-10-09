#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/backbone_torsion/potentials/backbone_torsion_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace backbone_torsion {
namespace potentials {

template struct BackboneTorsionPoseScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct BackboneTorsionPoseScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
