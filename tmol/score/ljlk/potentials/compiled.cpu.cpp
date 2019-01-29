#include <tmol/score/common/dispatch.cpu.impl.hh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define declare_dispatch(Real, Int)                                            \
  template struct LJDispatch<NaiveDispatch, tmol::Device::CPU, Real, Int>;     \
  template struct LJDispatch<NaiveTriuDispatch, tmol::Device::CPU, Real, Int>; \
  template struct LJDispatch<                                                  \
      ExhaustiveDispatch,                                                      \
      tmol::Device::CPU,                                                       \
      Real,                                                                    \
      Int>;                                                                    \
  template struct LJDispatch<                                                  \
      ExhaustiveTriuDispatch,                                                  \
      tmol::Device::CPU,                                                       \
      Real,                                                                    \
      Int>;                                                                    \
                                                                               \
  template struct LKIsotropicDispatch<                                         \
      NaiveDispatch,                                                           \
      tmol::Device::CPU,                                                       \
      Real,                                                                    \
      Int>;                                                                    \
  template struct LKIsotropicDispatch<                                         \
      NaiveTriuDispatch,                                                       \
      tmol::Device::CPU,                                                       \
      Real,                                                                    \
      Int>;                                                                    \
  template struct LKIsotropicDispatch<                                         \
      ExhaustiveDispatch,                                                      \
      tmol::Device::CPU,                                                       \
      Real,                                                                    \
      Int>;                                                                    \
  template struct LKIsotropicDispatch<                                         \
      ExhaustiveTriuDispatch,                                                  \
      tmol::Device::CPU,                                                       \
      Real,                                                                    \
      Int>;

declare_dispatch(float, int32_t);
declare_dispatch(double, int32_t);
declare_dispatch(float, int64_t);
declare_dispatch(double, int64_t);

#undef declare_dispatch

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
