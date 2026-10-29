// Instantiations of the hbond dispatch for the Metal backend.
//
// Only float is instantiated: Metal has no double, so a float64 pose cannot
// be scored on the GPU and is rejected by the dispatch macro's type switch
// before reaching here. The polynomial coefficients are the exception -- they
// need more than float32 precision and carry it as a hi/lo float32 pair; see
// params_types.hh.

#include <tmol/score/common/device_operations.mps.host.impl.hh>
#include <tmol/score/hbond/potentials/hbond_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template struct HBondPoseScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;

template struct HBondRotamerScoreDispatch<
    DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
