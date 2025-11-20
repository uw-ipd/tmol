#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/ljlk/potentials/ljlk_fusion_module.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJLKPoseScoreFusionModule<
    DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct LJLKPoseScoreFusionModule<
    DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
