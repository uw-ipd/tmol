#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/omega/potentials/omega_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template struct OmegaPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct OmegaPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
