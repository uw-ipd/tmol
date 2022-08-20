#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>

#include "apsp.hh"

namespace tmol {
namespace pose {

using torch::Tensor;

Tensor apsp_op(
    Tensor stacked_distances
) {
  using tmol::utility::connect_backward_pass;

  at::Tensor coords;
  at::Tensor HTs;

  using Int = int32_t;

  TMOL_DISPATCH_INDEX_DEVICE(stacked_distances.type(), "stacked_apsp_op", ([&] {
    using Int = scalar_t;
    constexpr tmol::Device Dev = device_t;
    
    AllPairsShortestPathsDispatch<Dev, Int>::f(TCAST(stacked_distances));
  }));
};


static auto registry = torch::jit::RegisterOperators()
  .op("tmol::apsp_op", &apsp_op);


}  // namespace pose
}  // namespace tmol
