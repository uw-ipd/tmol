#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/genbonded/potentials/genbonded_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace genbonded {
namespace potentials {

template struct GenBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct GenBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

template struct GenBondedRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct GenBondedRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
