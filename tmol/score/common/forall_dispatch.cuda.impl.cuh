#pragma once

#include <moderngpu/transform.hxx>

#include "forall_dispatch.hh"

namespace tmol {
namespace score {
namespace common {

template <>
struct ForallDispatch<tmol::Device::CUDA> {
  template <typename Int, typename Func>
  static void forall(Int N, Func f) {
    mgpu::standard_context_t context;
    mgpu::transform(f, N, context);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
