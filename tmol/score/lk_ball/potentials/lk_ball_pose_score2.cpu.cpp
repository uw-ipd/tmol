#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_pose_score2.impl.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct LKBallPoseScoreDispatch2<
    common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct LKBallPoseScoreDispatch2<
    common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
