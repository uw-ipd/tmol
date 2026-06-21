#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters.impl.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct GeneratePoseWaters<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct GeneratePoseWaters<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
