#include <tmol/score/common/dispatch.cuda.impl.cuh>

#include "dispatch.impl.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

#define declare_dispatch(Real, Int)                                         \
  template struct LJDispatch<NaiveDispatch, tmol::Device::CUDA, Real, Int>; \
  template struct LJDispatch<                                               \
      NaiveTriuDispatch,                                                    \
      tmol::Device::CUDA,                                                   \
      Real,                                                                 \
      Int>;                                                                 \
  template struct LJDispatch<                                               \
      ExhaustiveDispatch,                                                   \
      tmol::Device::CUDA,                                                   \
      Real,                                                                 \
      Int>;                                                                 \
  template struct LJDispatch<                                               \
      ExhaustiveTriuDispatch,                                               \
      tmol::Device::CUDA,                                                   \
      Real,                                                                 \
      Int>;                                                                 \
                                                                            \
  template struct LKIsotropicDispatch<                                      \
      NaiveDispatch,                                                        \
      tmol::Device::CUDA,                                                   \
      Real,                                                                 \
      Int>;                                                                 \
  template struct LKIsotropicDispatch<                                      \
      NaiveTriuDispatch,                                                    \
      tmol::Device::CUDA,                                                   \
      Real,                                                                 \
      Int>;                                                                 \
  template struct LKIsotropicDispatch<                                      \
      ExhaustiveDispatch,                                                   \
      tmol::Device::CUDA,                                                   \
      Real,                                                                 \
      Int>;                                                                 \
  template struct LKIsotropicDispatch<                                      \
      ExhaustiveTriuDispatch,                                               \
      tmol::Device::CUDA,                                                   \
      Real,                                                                 \
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
