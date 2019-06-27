#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template struct DunbrackDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int32_t,
    2,
    4>;
template struct DunbrackDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int32_t,
    2,
    4>;

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
