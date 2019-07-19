#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template struct DunbrackDispatch<common::ForallDispatch,tmol::Device::CPU,float,int32_t,2,4>;
template struct DunbrackDispatch<common::ForallDispatch,tmol::Device::CPU,double,int32_t,2,4>;


}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
