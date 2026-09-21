#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/ljlk/potentials/ljlk_elec_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJLKAndElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct LJLKAndElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
