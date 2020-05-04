#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template struct RamaDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int32_t>;
template struct RamaDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int32_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
