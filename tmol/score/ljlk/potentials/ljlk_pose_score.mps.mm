#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJLKPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct LJLKPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

template struct LJLKRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct LJLKRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
