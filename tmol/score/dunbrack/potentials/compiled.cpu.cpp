#include <tmol/score/common/forall_dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

template struct DunbrackDispatch<common::ForallDispatch,tmol::Device::CPU,float,int32_t>;
template struct DunbrackDispatch<common::ForallDispatch,tmol::Device::CPU,double,int32_t>;


}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
