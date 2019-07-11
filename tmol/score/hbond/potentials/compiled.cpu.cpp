#include <tmol/score/common/forall_dispatch.cpu.impl.hh>
#include <tmol/score/common/simple_dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {
#define declare_dispatch(Real, Int)                                          \
  template struct HBondDispatch<ForallDispatch, AABBDispatch, tmol::Device::CPU, Real, Int>; \
  template struct HBondDispatch<ForallDispatch, AABBTriuDispatch, tmol::Device::CPU, Real, Int>;

declare_dispatch(float, int32_t);
declare_dispatch(double, int32_t);
declare_dispatch(float, int64_t);
declare_dispatch(double, int64_t);

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
