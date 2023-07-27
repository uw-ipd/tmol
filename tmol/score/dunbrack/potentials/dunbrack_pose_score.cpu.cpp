#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/dunbrack/potentials/dunbrack_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template struct DunbrackPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct DunbrackPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
