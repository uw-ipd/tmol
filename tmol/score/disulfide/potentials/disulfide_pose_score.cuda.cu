#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/disulfide/potentials/disulfide_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template struct DisulfidePoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct DisulfidePoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
