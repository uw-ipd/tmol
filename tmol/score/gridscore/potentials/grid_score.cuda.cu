#include <tmol/score/common/device_operations.cuda.impl.cuh>
#include <tmol/score/gridscore/potentials/grid_score.impl.hh>

namespace tmol {
namespace score {
namespace gridscore {
namespace potentials {

template struct GridScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct GridScoreDispatch<
    common::DeviceOperations,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace gridscore
}  // namespace score
}  // namespace tmol
