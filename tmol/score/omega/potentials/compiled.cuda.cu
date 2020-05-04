#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template struct OmegaDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int32_t>;
template struct OmegaDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int32_t>;

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
