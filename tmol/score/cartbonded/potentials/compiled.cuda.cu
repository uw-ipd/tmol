#include <Eigen/Core>

#include <tmol/score/common/forall_dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template struct CartBondedDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    float,
    int64_t>;
template struct CartBondedDispatch<
    common::ForallDispatch,
    tmol::Device::CUDA,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
