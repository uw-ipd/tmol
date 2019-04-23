#include <torch/script.h>
#include <iostream>
#include <tmol/utility/autograd.hh>

struct CPowBackward : public torch::autograd::Function {
  // CPow backward pass autograd function.
  //
  // Backward pass functions can hold references to tensor data via
  // SavedVariable members and can also include arbitrary member data.
  //
  // SavedVariable members are preferred as they allow control of tensor data
  // lifetime (eg. calls to `backward(retain_graph=True)`) and must be managed
  // via the `release_variables` member function.
  torch::autograd::SavedVariable saved_x;
  double exponent;

  void release_variables() override {
    saved_x.reset_data();
    saved_x.reset_grad_function();
  }

  CPowBackward(torch::autograd::Variable x, double exponent)
      : saved_x(x, false), exponent(exponent), torch::autograd::Function() {}

  // Note that autograd functions are of the form [input]->[output], operating
  // on positional lists of inputs and outputs rather than standard function
  // signatures. The positional association between inputs and outputs is
  // determined by the order of attachment in `connect_backward_pass`
  //
  // Function::apply receives an input for every member of the connected
  // forward `outputs` and must return an output for every member of the
  // connected forward `inputs`.
  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    auto x = saved_x.unpack();

    auto& d_r = grads[0];

    // Unconditionally evaluate the gradient, but could optimize via
    // should_compute_output(0) if only some gradients are required.
    auto d_x = exponent != 0.0 ? d_r * exponent * x.pow(exponent - 1)
                               : torch::zeros_like(x);

    return {d_x};
  }
};

// Forward pass, exposed as a TorchScript custom op.
//
// Op calculates the forward result as a detached tensor, then integrates
// autograd via `connect_backward_pass`.
torch::Tensor cpow(torch::Tensor x, double exponent) {
  torch::Tensor result = x.detach().pow(exponent);

  using tmol::utility::connect_backward_pass;

  return connect_backward_pass({x}, result, [&]() {
    return std::shared_ptr<CPowBackward>(
        new CPowBackward(x, exponent), torch::autograd::deleteFunction);
  });
}

static auto registry = torch::jit::RegisterOperators("tmol::cpow", &cpow);
