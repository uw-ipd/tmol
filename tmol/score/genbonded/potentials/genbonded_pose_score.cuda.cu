#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/genbonded/potentials/genbonded_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace genbonded {
namespace potentials {

template struct GenBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct GenBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

template struct GenBondedRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct GenBondedRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
