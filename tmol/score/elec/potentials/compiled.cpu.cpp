#include <tmol/score/common/dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace elec {
namespace potentials {
#define declare_dispatch(Real, Int)                                          \
  template struct ElecDispatch<NaiveDispatch, tmol::Device::CPU, Real, Int>; \
  template struct ElecDispatch<NaiveTriuDispatch, tmol::Device::CPU, Real, Int>;

declare_dispatch(float, int32_t);
declare_dispatch(double, int32_t);
declare_dispatch(float, int64_t);
declare_dispatch(double, int64_t);

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
