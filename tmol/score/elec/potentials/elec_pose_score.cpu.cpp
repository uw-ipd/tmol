#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/elec/potentials/elec_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template struct ElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct ElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
