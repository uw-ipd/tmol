#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/disulfide/potentials/disulfide_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template struct DisulfidePoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct DisulfidePoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
