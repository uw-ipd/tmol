#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/hbond/potentials/hbond_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template struct HBondPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct HBondPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

template struct HBondPoseScoreDispatch2<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct HBondPoseScoreDispatch2<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
