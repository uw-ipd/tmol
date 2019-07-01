#pragma once

#include <Eigen/Core>

#include <ATen/cuda/CUDAStream.h>
#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ForallDispatch {
  template <typename Int, typename Func>
  void forall(Int N, Func f);

  template <typename Int, typename Func>
  void forall(Int N, Func f, at::cuda::CUDAStream stream);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
