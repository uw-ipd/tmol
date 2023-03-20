#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/rama/potentials/rama_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template struct RamaPoseScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct RamaPoseScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
