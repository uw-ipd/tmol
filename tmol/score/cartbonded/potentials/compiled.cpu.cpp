#include <Eigen/Core>

#include <tmol/score/common/forall_dispatch.cpu.impl.hh>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/score/common/geom.hh>

#include "dispatch.impl.hh"
#include "params.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template struct CartBondedLengthDispatch<common::ForallDispatch, tmol::Device::CPU, float, int64_t>;
template struct CartBondedLengthDispatch<common::ForallDispatch, tmol::Device::CPU, double, int64_t>;
template struct CartBondedAngleDispatch<common::ForallDispatch, tmol::Device::CPU, float, int64_t>;
template struct CartBondedAngleDispatch<common::ForallDispatch, tmol::Device::CPU, double, int64_t>;
template struct CartBondedTorsionDispatch<common::ForallDispatch, tmol::Device::CPU, float, int64_t>;
template struct CartBondedTorsionDispatch<common::ForallDispatch, tmol::Device::CPU, double, int64_t>;
template struct CartBondedHxlTorsionDispatch<common::ForallDispatch, tmol::Device::CPU, float, int64_t>;
template struct CartBondedHxlTorsionDispatch<
    common::ForallDispatch, 
    tmol::Device::CPU,
    double,
    int64_t>;

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
