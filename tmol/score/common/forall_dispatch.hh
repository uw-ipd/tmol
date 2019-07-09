#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/cuda/stream.hh>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ForallDispatch {
  template <typename Int, typename Func>
  void forall(Int N, Func f, utility::cuda::CUDAStream stream);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
