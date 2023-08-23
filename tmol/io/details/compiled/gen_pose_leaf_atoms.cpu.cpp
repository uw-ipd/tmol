#include <tmol/score/common/device_operations.cpu.impl.hh>
#include <tmol/io/details/compiled/gen_pose_leaf_atoms.impl.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    float,
    int>;
template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CPU,
    double,
    int>;

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
