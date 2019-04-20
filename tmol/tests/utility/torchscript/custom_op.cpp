#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/script.h>

struct CPowBackward : public torch::autograd::Function {
  torch::autograd::SavedVariable saved_x;
  double exponent;

  CPowBackward(torch::Tensor x, double exponent)
      : Function(), saved_x(x, false), exponent(exponent) {}

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    auto& dx = grads[0];
    auto x = saved_x.unpack();

    torch::autograd::variable_list input_grads(1);

    if (should_compute_output(0)) {
      input_grads[0] = exponent != 0.0 ? dx * exponent * x.pow(exponent - 1)
                                       : torch::zeros_like(x);
    }

    return input_grads;
  }

  void release_variables() override {
    saved_x.reset_data();
    saved_x.reset_grad_function();
  }
};

torch::Tensor CPow(torch::Tensor x, double exponent) {
  // Compute the function's output
  auto result = x.detach().pow(exponent);

  if (torch::autograd::any_variable_requires_grad({x})) {
    auto backward = std::shared_ptr<CPowBackward>(
        new CPowBackward(x, exponent), torch::autograd::deleteFunction);

    // Connect into the autograd graph
    backward->set_next_edges(torch::autograd::collect_next_edges(x));

    torch::autograd::create_gradient_edge(
        torch::autograd::as_variable_ref(result), backward);
  }

  return result;
}

static auto registry = torch::jit::RegisterOperators("tmol::CPow", &CPow);
