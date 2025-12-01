#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/elec/potentials/elec_fusion_module.impl.hh>

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

template struct ElecPoseScoreFusionModule<
    DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct ElecPoseScoreFusionModule<
    DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
