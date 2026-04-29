#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include "lbfgs.hh"

namespace tmol {
namespace optimization {
namespace compiled {

using torch::Tensor;

Tensor lbfgs_two_loop(Tensor grad, Tensor dirs, Tensor stps) {
  TORCH_CHECK(grad.is_contiguous());
  TORCH_CHECK(dirs.is_contiguous());
  TORCH_CHECK(stps.is_contiguous());

  auto ro = (dirs * stps).sum(1).reciprocal_();

  Tensor out;
  TMOL_DISPATCH_FLOATING_DEVICE(
      grad.options(), "lbfgs_two_loop", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result = LbfgsTwoLoop<Dev, Real>::f(
            TCAST(grad), TCAST(dirs), TCAST(stps), TCAST(ro));
        out = result.tensor;
      }));
  return out;
}

TORCH_LIBRARY(tmol_optimization, m) {
  m.def("lbfgs_two_loop", &lbfgs_two_loop);
}

}  // namespace compiled
}  // namespace optimization
}  // namespace tmol
