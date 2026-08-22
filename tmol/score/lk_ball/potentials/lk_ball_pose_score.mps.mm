#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/lk_ball/potentials/lk_ball_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct LKBallPoseScoreDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct LKBallPoseScoreDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

template struct LKBallRotamerScoreDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct LKBallRotamerScoreDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
