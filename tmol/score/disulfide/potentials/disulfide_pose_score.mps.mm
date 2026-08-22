#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/disulfide/potentials/disulfide_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace disulfide {
namespace potentials {

template struct DisulfidePoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct DisulfidePoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

template struct DisulfideRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct DisulfideRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

}  // namespace potentials
}  // namespace disulfide
}  // namespace score
}  // namespace tmol
