#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include <tmol/score/rama/potentials/params.hh>
#include <tmol/score/rama/potentials/dispatch.hh>

namespace tmol {
namespace score {
namespace rama {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class ScoreOp : public torch::autograd::Function<ScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor params,
      Tensor tables,
      Tensor table_params) {
    at::Tensor score;
    at::Tensor dScore;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "rama_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          // std::cout << "Rama Score Op: " << std::endl;

          auto result = RamaDispatch<DispatchMethod, Dev, Real, Int>::f(
              TCAST(coords), TCAST(params), TCAST(tables), TCAST(table_params));

          score = std::get<0>(result).tensor;
          dScore = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({dScore});
    return score;
  }
  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    // std::cout << "Rama backward" << std::endl;
    auto saved_grads = ctx->get_saved_variables();

    tensor_list result;

    for (auto& saved_grad : saved_grads) {
      auto ingrad = grad_outputs[0];
      while (ingrad.dim() < saved_grad.dim()) {
        ingrad = ingrad.unsqueeze(-1);
      }

      result.emplace_back(saved_grad * ingrad);
    }

    int i = 0;
    auto dCoords = result[i++];

    return {
        dCoords,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor score_op(
    Tensor coords, Tensor params, Tensor tables, Tensor table_params) {
  return ScoreOp<DispatchMethod>::apply(coords, params, tables, table_params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_rama", &score_op<common::ForallDispatch>);
}

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
