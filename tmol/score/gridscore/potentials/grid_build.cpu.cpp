#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/score/gridscore/potentials/grid_build.impl.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template struct GridBuildDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int64_t>;

template struct GridBuildDispatch<
    common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int64_t>;

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
