#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/forall_dispatch.hh>

#include <pybind11/pybind11.h>

#include "params.hh"
#include "dispatch.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

using torch::Tensor;


struct ScoreOpBackward : public torch::autograd::Function {
  typedef torch::autograd::Variable Variable;
  typedef torch::autograd::SavedVariable SavedVariable;
  typedef torch::autograd::variable_list variable_list;

  static std::shared_ptr<ScoreOpBackward> create(variable_list&& grads) {
    return std::shared_ptr<ScoreOpBackward>(
        new ScoreOpBackward(std::move(grads)),
        torch::autograd::deleteFunction);
  }

  std::vector<SavedVariable> saved_grads;

  void release_variables() override {
    for (auto& saved_grad : saved_grads) {
      saved_grad.reset_data();
      saved_grad.reset_grad_function();
    }
  }

  ScoreOpBackward(variable_list&& grads) : torch::autograd::Function() {
    saved_grads.reserve(grads.size());

    for (auto& grad : grads) {
      saved_grads.emplace_back(SavedVariable(grad, false));
    }
  }

  variable_list apply(variable_list&& in_grads) override {
    variable_list result;
    result.reserve(saved_grads.size());

    for (auto& saved_grad : saved_grads) {
      auto x = saved_grad.unpack();
      result.emplace_back( (x * in_grads[0].unsqueeze(1)).sum(1) );
    }

    return result;
  }
};


// The op for cartbonded dispatch
// Uses abbreviations:
//   cbl = cartbonded_length
//   cba = cartbonded_angle
//   cbt = cartbonded_torsion
//   cbi = cartbonded_improper_torsion
//   cbhxl = cartbonded_hydroxyl_torsion
template < template <tmol::Device> class DispatchMethod >
Tensor cb_score_op(
      Tensor coords,
      Tensor cbl_atoms,
      Tensor cba_atoms,
      Tensor cbt_atoms,
      Tensor cbi_atoms,
      Tensor cbhxl_atoms,
      Tensor cbl_params,
      Tensor cba_params,
      Tensor cbt_params,
      Tensor cbi_params,
      Tensor cbhxl_params
) {
  using tmol::utility::connect_backward_pass;
  nvtx_range_push("cb_score_op");

  at::Tensor score;
  at::Tensor dScore;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      coords.type(), "cb_score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = CartBondedDispatch<DispatchMethod, Dev, Real, Int>::f(
            TCAST(coords),
            TCAST(cbl_atoms),
            TCAST(cba_atoms),
            TCAST(cbt_atoms),
            TCAST(cbi_atoms),
            TCAST(cbhxl_atoms),
            TCAST(cbl_params),
            TCAST(cba_params),
            TCAST(cbt_params),
            TCAST(cbi_params),
            TCAST(cbhxl_params));

        score = std::get<0>(result).tensor;
        dScore = std::get<1>(result).tensor;
      }));

  auto backward_op = connect_backward_pass({coords}, score, [&]() {
    return ScoreOpBackward::create({dScore});
  });

  nvtx_range_pop();
  return backward_op;
};

static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_cartbonded", &cb_score_op<common::ForallDispatch>);

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
