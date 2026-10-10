#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/density/potentials/density_score.impl.hh>

namespace tmol {
namespace score {
namespace density {
namespace potentials {

template struct DensityScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float>;
template struct DensityScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double>;

}  // namespace potentials
}  // namespace density
}  // namespace score
}  // namespace tmol
