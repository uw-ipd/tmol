#include <tmol/score/common/device_operations.mps.impl.hh>
#include <tmol/score/elec/potentials/elec_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template struct ElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct ElecPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

template struct ElecRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;
template struct ElecRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    double,
    int>;

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
