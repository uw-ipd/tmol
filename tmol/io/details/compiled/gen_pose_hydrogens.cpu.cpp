#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/io/details/compiled/gen_pose_hydrogens.impl.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template struct GeneratePoseHydrogens<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct GeneratePoseHydrogens<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
