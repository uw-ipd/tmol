#include <Eigen/Core>

#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template struct CartBondedDispatch<
    common::ForallDispatch,
    tmol::Device::CPU,
    float,
    int64_t>;
template struct CartBondedDispatch<
    common::ForallDispatch,
    tmol::Device::CPU,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
