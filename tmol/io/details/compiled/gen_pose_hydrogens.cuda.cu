#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/io/details/compiled/gen_pose_hydrogens.impl.hh>

namespace tmol {
namespace io {
namespace details {
namespace compiled {

template struct GeneratePoseHydrogens<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int>;
template struct GeneratePoseHydrogens<
    score::common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int>;

}  // namespace compiled
}  // namespace details
}  // namespace io
}  // namespace tmol
