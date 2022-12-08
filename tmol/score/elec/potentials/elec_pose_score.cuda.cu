#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/elec/potentials/elec_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template struct ElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct ElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
