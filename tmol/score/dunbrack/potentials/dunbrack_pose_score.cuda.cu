#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/dunbrack/potentials/dunbrack_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template struct DunbrackPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct DunbrackPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
