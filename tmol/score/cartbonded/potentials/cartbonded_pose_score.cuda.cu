#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template struct CartBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct CartBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
