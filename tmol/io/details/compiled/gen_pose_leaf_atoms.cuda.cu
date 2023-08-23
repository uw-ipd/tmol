#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/io/details/compiled/gen_pose_leaf_atoms.impl.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct GeneratePoseLeafAtoms<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
