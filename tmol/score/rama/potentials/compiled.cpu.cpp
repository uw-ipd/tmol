#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

template struct RamaDispatch<
    common::ForallDispatch,
    tmol::Device::CPU,
    float,
    int32_t>;
template struct RamaDispatch<
    common::ForallDispatch,
    tmol::Device::CPU,
    double,
    int32_t>;

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
