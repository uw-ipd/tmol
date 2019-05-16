#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

template struct OmegaDispatch<common::ForallDispatch, tmol::Device::CPU,float,int32_t>;
template struct OmegaDispatch<common::ForallDispatch, tmol::Device::CPU,double,int32_t>;

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
