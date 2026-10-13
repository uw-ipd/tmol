#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/density/potentials/density_score.impl.hh>

namespace tmol {
namespace score {
namespace density {
namespace potentials {

template struct DensityScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    float>;
template struct DensityScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    double>;

}  // namespace potentials
}  // namespace density
}  // namespace score
}  // namespace tmol
