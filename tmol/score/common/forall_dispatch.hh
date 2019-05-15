#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>

namespace tmol {
namespace score {
namespace common {

template <tmol::Device D>
struct ForallDispatch {
  template <typename Int, typename Func>
  void forall(Int N, Func f);
};

}  // namespace common
}  // namespace score
}  // namespace tmol
