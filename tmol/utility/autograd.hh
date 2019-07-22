#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/types.h>

#include <pybind11/pybind11.h>

#include <tmol/utility/nvtx.hh>

namespace tmol {
namespace utility {

// Note the somewhat tricky interaction between `torch::Tensor` and
// `torch::autograd::Variable` types below.
//
// `Tensor` represents a arbitrary tensor value. `Variable` represents a
// `Tensor` with autograd integration, and may be in an attached (requires_grad
// and tracing) or detached (!requires_grad and not tracing) state.
//
// `Variable` is a subtype of `Tensor`, and the types are implicitly
// convertible. While all `Variable` are `Tensor`, not all `Tensor` are
// `Variable`. This results in three potential states for any `Tensor` value:
//
//   * `plain` - Just a `Tensor`, `!is_variable & !requires_grad`.
//   * `detached` - A `Variable` without tracing, `is_variable &
//   !requires_grad`.
//   * `attached` - A `Variable` with tracing, `is_variable & requires_grad`.
//
//  `plain` can be wrapped into a `detached` via `make_variable`,
//  `detached`/`attached` both expose an unwrapped `plain` via `data()`
//  `attached` can be demoted to `detached` via `detach()`.
//
// TorchScript ops always receive `Tensor` types in `attached` or `detached`
// state. They should return corresponding `attached` or `detached` results.
//
// If the op is purely composed of autograd-compatible calls on inputs there is
// no need to consider the backward pass. Execute the forward pass normally and
// allow autograd to handle the backward pass.
//
// If the op makes autograd-incompatible calls (eg. direct access to tensor
// values, calls to external functions...) then the op must initialize and
// attach a autograd graph, converting any `plain` or `detached` outputs into
// `attached` variables. This is handled by defining an `autograd::Function`
// for the backward pass and connecting it via `connect_backward_pass`.

// Adaptation of v1.0.1 torch::autograd::wrap_outputs in
// torch/csrc/autograd/functions/utils.cpp
//
// Connects torch::autograd backward pass for `inputs` and `outputs` of a
// evaluated forward pass iff gradients are required for any `inputs`.
//
// Params:
//   inputs:
//     Forward inputs, must be `Variable` but may be detached.  Backward pass
//     attachment will be skipped if no inputs require_grad.
//
//   outputs:
//     Forward outputs, may be Variable *or* plain Tensor. All outputs are
//     promoted to Variable, but will be detached if no inputs require_grad.
//
//   backward_factory:
//     Factory method for backward-pass autograd::Function.  Function::apply
//     must accept gradient inputs matching `outputs` and return gradient
//     outputs matching `inputs`. Factory may capture arbitrary inputs or
//     intermediate tensors as SavedVars. Autograd edges next and gradient
//     edges will be configured post-construction by `connect_backward_pass`.
//
// Returns:
//   A variable_list matching `outputs` which must be returned in place of
//   `outputs` from the forward operation.
template <typename function_factory>
torch::autograd::variable_list connect_backward_pass(
    const torch::autograd::variable_list& inputs,
    torch::autograd::tensor_list&& outputs,
    function_factory backward_factory) {
  auto as_output_var = [](auto output) -> torch::autograd::Variable {
    if (output.defined()) {
      if (output.is_variable()) {
        AT_CHECK(
            !output.requires_grad(),
            "Can't connect gradient Function, output already requires_grad.");
        return torch::autograd::as_variable_ref(output);
      } else {
        return torch::autograd::make_variable(output, /*requires_grad=*/false);
      }
    } else {
      return torch::autograd::Variable();
    }
  };

  torch::autograd::variable_list result;
  result.reserve(outputs.size());

  if (!any_variable_requires_grad(inputs)) {
    // No gradient required, setup detached output variables.
    for (auto& output : outputs) {
      result.push_back(as_output_var(output));
    }
  } else {
    // Gradient required, setup output variables and attach via backward.

    // 1) Collect input variable edges before calling factory, ensuring that
    // input vars have autograd edges attached and can be wrapped in SavedVars
    // within function factory.
    auto edges = collect_next_edges(inputs);

    // 2) Construct the gradient function and connect input edges to outputs of
    // the backward pass.
    std::shared_ptr<torch::autograd::Function> grad_fn = backward_factory();
    grad_fn->set_next_edges(std::move(edges));

    // 2) Setup output variables and attach gradient edges to inputs of the
    // backward pass.
    for (auto& output : outputs) {
      auto variable = as_output_var(output);

      if (variable.defined()) {
        torch::autograd::create_gradient_edge(variable, grad_fn);
      } else {
        grad_fn->add_input_metadata(
            torch::autograd::Function::undefined_input());
      }

      result.push_back(std::move(variable));
    }
  }
  return result;
}

// Utility wrapper for connect_backward_pass one output.
template <typename function_factory>
torch::autograd::Variable connect_backward_pass(
    const torch::autograd::variable_list& inputs,
    torch::Tensor& output,
    function_factory backward_factory) {
  nvtx_range_push("connect_backward_pass");
  torch::autograd::tensor_list outputs = {output};
  auto bw = connect_backward_pass(inputs, std::move(outputs), backward_factory)
                .front();
  nvtx_range_pop();
  return bw;
}

// Support function for backward-pass through pre-calculated gradients.
//
// Returns dT_dR * dR_dI_n for n precalculated gradients where
// "T" is the total score, R is the returned tensors, and
// "I" are the input tensors, and where where dT_dR is a
// one-dimensional tensor of per-stack gradients. The apply
// function performs a broadcasting multiplication of the
// trailing dimensions of the tensors.
struct StackedSavedGradsBackward : public torch::autograd::Function {
  typedef torch::autograd::Variable Variable;
  typedef torch::autograd::SavedVariable SavedVariable;
  typedef torch::autograd::variable_list variable_list;

  // Factory method, for use with connect_backward_pass.
  static std::shared_ptr<StackedSavedGradsBackward> create(variable_list&& grads) {
    return std::shared_ptr<StackedSavedGradsBackward>(
        new StackedSavedGradsBackward(std::move(grads)),
        torch::autograd::deleteFunction);
  }

  std::vector<SavedVariable> saved_grads;

  void release_variables() override {
    for (auto& saved_grad : saved_grads) {
      saved_grad.reset_data();
      saved_grad.reset_grad_function();
    }
  }

  StackedSavedGradsBackward(variable_list&& grads) : torch::autograd::Function() {
    saved_grads.reserve(grads.size());

    for (auto& grad : grads) {
      saved_grads.emplace_back(SavedVariable(grad, false));
    }
  }

  variable_list apply(variable_list&& in_grads) override {
    NVTXRange("StackedSavedGradsBackward");

    AT_CHECK(
        in_grads.size() == 1,
        "StackedSavedGradsBackward only supports a single gradient input");
    for (auto& saved_grad : saved_grads) {
      AT_CHECK(
          in_grads[0].size(0) == saved_grad.unpack().size(0)
              || in_grads[0].size(0) == 1,
          "Tensors sizes must match along the first dimension, the stack dimension");
    }

    variable_list result;
    result.reserve(saved_grads.size());

    for (auto& saved_grad : saved_grads) {
      auto x = saved_grad.unpack();
      std::vector<int64_t> newdims(x.dim(), 1);
      newdims[0] = in_grads[0].size(0);
      c10::IntList newdims_il(&newdims[0], x.dim());
      auto ingrad = in_grads[0].view(newdims_il);
      result.emplace_back(saved_grad.unpack() * ingrad);
    }

    nvtx_range_pop();
    return result;
  }
};

}  // namespace utility
}  // namespace tmol
