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
    int32_t>;

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
