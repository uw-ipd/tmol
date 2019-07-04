#include <tmol/score/common/simple_dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

#define declare_dispatch(Real, Int)                                          \
  template struct ElecDispatch<AABBDispatch, tmol::Device::CUDA, Real, Int>; \
  template struct ElecDispatch<AABBTriuDispatch, tmol::Device::CUDA, Real, Int>;

declare_dispatch(float, int64_t);
declare_dispatch(double, int64_t);

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
