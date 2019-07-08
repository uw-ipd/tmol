#include <tmol/score/common/forall_dispatch.cpu.impl.hh>
#include <tmol/score/common/simple_dispatch.cpu.impl.hh>

#include "lk_isotropic.dispatch.impl.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define declare_dispatch(Real, Int)                                            \
  template struct LKIsotropicDispatch<ForallDispatch, AABBDispatch, tmol::Device::CPU, Real, Int>; \
  template struct LKIsotropicDispatch<ForallDispatch, AABBTriuDispatch, tmol::Device::CPU, Real, Int>; \

declare_dispatch(float, int64_t);
declare_dispatch(double, int64_t);

#undef declare_dispatch

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
