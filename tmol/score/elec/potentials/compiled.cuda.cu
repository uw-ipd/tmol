#include <tmol/score/common/dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace elec {
namespace potentials {
#define declare_dispatch(Real, Int)                                           \
  template struct ElecDispatch<NaiveDispatch, tmol::Device::CUDA, Real, Int>; \
  template struct ElecDispatch<                                               \
      NaiveTriuDispatch,                                                      \
      tmol::Device::CUDA,                                                     \
      Real,                                                                   \
      Int>;

declare_dispatch(float, int32_t);
declare_dispatch(double, int32_t);

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
