// Instantiations of the LJ/LK dispatch for the Metal backend.
//
// Only float is instantiated: Metal has no double, so a float64 pose cannot
// be scored on the GPU and is rejected by the dispatch macro's type switch
// before reaching here.

#include <tmol/score/common/device_operations.mps.host.impl.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template struct LJLKPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;

template struct LJLKRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
