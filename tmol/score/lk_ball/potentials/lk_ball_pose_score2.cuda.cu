#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/lk_ball/potentials/lk_ball_pose_score2.impl.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct LKBallPoseScoreDispatch2<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct LKBallPoseScoreDispatch2<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
