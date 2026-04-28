#pragma once

#include <tmol/utility/tensor/TensorPack.h>

namespace tmol {
namespace optimization {
namespace compiled {

template <tmol::Device D, typename Real>
struct LbfgsTwoLoop {
  static auto f(
      TView<Real, 1, D> grad,
      TView<Real, 2, D> dirs,
      TView<Real, 2, D> stps,
      TView<Real, 1, D> ro) -> TPack<Real, 1, D>;
};

}  // namespace compiled
}  // namespace optimization
}  // namespace tmol
