#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/lk_ball/potentials/gen_pose_waters.impl.hh>

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

template struct GeneratePoseWaters<
    common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct GeneratePoseWaters<
    common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
