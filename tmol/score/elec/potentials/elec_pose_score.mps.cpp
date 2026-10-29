// Instantiations of the elec dispatch for the Metal backend.
//
// Only float is instantiated: Metal has no double, so a float64 pose cannot
// be scored on the GPU and is rejected by the dispatch macro's type switch
// before reaching here.

#include <tmol/score/common/device_operations.mps.host.impl.hh>
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

template struct ElecRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
