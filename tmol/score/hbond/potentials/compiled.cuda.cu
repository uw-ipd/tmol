#include <tmol/score/common/dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {
#define declare_dispatch(Real, Int)                                            \
  template struct HBondDispatch<NaiveDispatch, tmol::Device::CUDA, Real, Int>; \
  template struct HBondDispatch<                                               \
      NaiveTriuDispatch,                                                       \
      tmol::Device::CUDA,                                                      \
      Real,                                                                    \
      Int>;                                                                    \
  template struct HBondDispatch<                                               \
      ExhaustiveDispatch,                                                      \
      tmol::Device::CUDA,                                                      \
      Real,                                                                    \
      Int>;                                                                    \
  template struct HBondDispatch<                                               \
      ExhaustiveTriuDispatch,                                                  \
      tmol::Device::CUDA,                                                      \
      Real,                                                                    \
      Int>;

declare_dispatch(float, int32_t);
declare_dispatch(double, int32_t);

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
