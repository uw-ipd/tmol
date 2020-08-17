#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include "dispatch.hh"
#include <tmol/utility/nvtx.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
class ScoreOp
    : public torch::autograd::Function<ScoreOp<ScoreDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor donor_coords,
      Tensor acceptor_coords,
      Tensor Dinds,
      Tensor H,
      Tensor donor_type,
      Tensor A,
      Tensor B,
      Tensor B0,
      Tensor acceptor_type,
      Tensor pair_params,
      Tensor pair_polynomials,
      Tensor global_params) {
    at::Tensor score;
    at::Tensor dV_d_don;
    at::Tensor dV_d_acc;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        donor_coords.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = HBondDispatch<DispatchMethod, Dev, Real, Int>::f(
              TCAST(donor_coords),
              TCAST(acceptor_coords),
              TCAST(Dinds),
              TCAST(H),
              TCAST(donor_type),
              TCAST(A),
              TCAST(B),
              TCAST(B0),
              TCAST(acceptor_type),
              TCAST(pair_params),
              TCAST(pair_polynomials),
              TCAST(global_params));

          score = std::get<0>(result).tensor;
          dV_d_don = std::get<1>(result).tensor;
          dV_d_acc = std::get<2>(result).tensor;
        }));

    ctx->save_for_backward({dV_d_don, dV_d_acc});

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
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
    auto dT_d_don = result[i++];
    auto dT_d_acc = result[i++];

    return {
        dT_d_don,
        dT_d_acc,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
  }
};

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
Tensor score_op(
    Tensor donor_coords,
    Tensor acceptor_coords,
    Tensor Dinds,
    Tensor H,
    Tensor donor_type,
    Tensor A,
    Tensor B,
    Tensor B0,
    Tensor acceptor_type,
    Tensor pair_params,
    Tensor pair_polynomials,
    Tensor global_params) {
  return ScoreOp<ScoreDispatch, DispatchMethod>::apply(
      donor_coords,
      acceptor_coords,
      Dinds,
      H,
      donor_type,
      A,
      B,
      B0,
      acceptor_type,
      pair_params,
      pair_polynomials,
      global_params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_hbond", &score_op<HBondDispatch, common::AABBDispatch>);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
