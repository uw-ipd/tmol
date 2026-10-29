// Instantiations of the LK-ball dispatch for the Metal backend.
//
// Only float is instantiated: Metal has no double, so a float64 pose cannot
// be scored on the GPU and is rejected by the dispatch macro's type switch
// before reaching here.

#include <tmol/score/common/device_operations.mps.host.impl.hh>
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

template struct LKBallRotamerScoreDispatch<
    common::DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
