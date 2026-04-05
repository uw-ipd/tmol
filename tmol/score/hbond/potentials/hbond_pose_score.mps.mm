#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/hbond/potentials/hbond_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template struct HBondPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct HBondPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

template struct HBondRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct HBondRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
