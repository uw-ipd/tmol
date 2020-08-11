#include <torch/script.h>
#include <iostream>

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class CPow : public Function<CPow> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor x, double exponent) {
    ctx->save_for_backward({x});
    ctx->saved_data["exponent"] = exponent;

    return x.detach().pow(exponent);
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grads) {
    auto x = ctx->get_saved_variables()[0];
    double exponent = ctx->saved_data["exponent"].toDouble();
    auto& d_r = grads[0];

    auto d_x = exponent != 0.0 ? d_r * exponent * x.pow(exponent - 1)
                               : torch::zeros_like(x);
    return {d_x, torch::Tensor()};
  }
};

// Forward pass, exposed as a TorchScript custom op.
//
// Op calculates the forward result as a detached tensor, then integrates
// autograd via `connect_backward_pass`.
torch::Tensor cpow(torch::Tensor x, double exponent) {
  return CPow::apply(x, exponent);
}

TORCH_LIBRARY(tmol, m) { m.def("cpow", &cpow); }
