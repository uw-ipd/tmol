
#pragma once

#include "forall_dispatch.hh"

namespace tmol {
namespace score {
namespace common {

template <>
struct ForallDispatch<tmol::Device::CPU> {

  template <typename Int, typename Func>
  static void forall(Int N, Func f, at::cuda::CUDAStream) {
    for (Int i = 0; i < N; ++i) {
      f(i);
    }
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
