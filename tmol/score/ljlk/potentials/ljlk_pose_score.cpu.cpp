#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJLKPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct LJLKPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
