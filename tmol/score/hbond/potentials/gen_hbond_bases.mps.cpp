// Instantiation of the hbond derived-atom pre-pass for the Metal backend.
//
// Only float is instantiated: Metal has no double, so a float64 pose cannot
// be scored on the GPU and is rejected by the dispatch macro's type switch
// before reaching here.

#include <tmol/score/common/device_operations.mps.host.impl.hh>
#include <tmol/score/hbond/potentials/gen_hbond_bases.impl.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

template struct GenerateHBondBases<
    common::DeviceOperations,
    tmol::Device::MPS,
    float,
    int>;

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
