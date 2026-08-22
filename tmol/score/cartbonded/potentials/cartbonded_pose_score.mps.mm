#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/cartbonded/potentials/cartbonded_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template struct CartBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct CartBondedPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

template struct CartBondedRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct CartBondedRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
