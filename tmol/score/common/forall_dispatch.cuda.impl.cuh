#pragma once

#include <moderngpu/transform.hxx>

#include "forall_dispatch.hh"
#include <tmol/utility/cuda/context.hh>

namespace tmol {
namespace score {
namespace common {

template <>
struct ForallDispatch<tmol::Device::CUDA> {

  template <typename Int, typename Func>
  static void forall(Int N, Func f, utility::cuda::CUDAStream stream) {
    mgpu::standard_context_t context = context_from_stream(stream);
    mgpu::transform(f, N, context);
  }
};

}  // namespace common
}  // namespace score
}  // namespace tmol
