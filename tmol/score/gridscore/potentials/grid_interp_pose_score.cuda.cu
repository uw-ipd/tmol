#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/gridscore/potentials/grid_interp_pose_score.impl.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template struct GridInterpDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct GridInterpDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
