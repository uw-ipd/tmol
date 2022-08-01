#include <torch/script.h>
#include <tmol/utility/autograd.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>

#include "compiled.hh"

namespace tmol {
namespace kinematics {

using torch::Tensor;

struct KinOpBackward : public torch::autograd::Function {
  torch::autograd::SavedVariable saved_hts;
  torch::autograd::SavedVariable saved_dofs;
  torch::autograd::SavedVariable saved_nodes;
  torch::autograd::SavedVariable saved_scans;
  torch::autograd::SavedVariable saved_gens;
  torch::autograd::SavedVariable saved_kintree;

  void release_variables() override {
    saved_hts.reset_data();
    saved_hts.reset_grad_function();
    saved_dofs.reset_data();
    saved_dofs.reset_grad_function();
    saved_nodes.reset_data();
    saved_nodes.reset_grad_function();
    saved_scans.reset_data();
    saved_scans.reset_grad_function();
    saved_gens.reset_data();
    saved_gens.reset_grad_function();
    saved_kintree.reset_data();
    saved_kintree.reset_grad_function();
  }

  KinOpBackward(
      torch::autograd::Variable hts,
      torch::autograd::Variable dofs,
      torch::autograd::Variable nodes,
      torch::autograd::Variable scans,
      torch::autograd::Variable gens,
      torch::autograd::Variable kintree)
      : saved_hts(hts, false),
        saved_dofs(dofs, false),
        saved_nodes(nodes, false),
        saved_scans(scans, false),
        saved_gens(gens, false),
        saved_kintree(kintree, false) {}

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    auto hts = saved_hts.unpack();
    auto dofs = saved_dofs.unpack();
    auto nodes = saved_nodes.unpack();
    auto scans = saved_scans.unpack();
    auto gens = saved_gens.unpack();
    auto kintree = saved_kintree.unpack();

    at::Tensor dV_ddof;
    using Int = int32_t;
    auto dVdx = grads[0];

    TMOL_DISPATCH_FLOATING_DEVICE(hts.type(), "kin_deriv_op", ([&] {
                                    using Real = scalar_t;
                                    constexpr tmol::Device Dev = device_t;

                                    auto result =
                                        KinDerivDispatch<Dev, Real, Int>::f(
                                            TCAST(dVdx),
                                            TCAST(hts),
                                            TCAST(dofs),
                                            TCAST(nodes),
                                            TCAST(scans),
                                            TCAST(gens),
                                            TCAST(kintree));

                                    dV_ddof = result.tensor;
                                  }));

    return {dV_ddof};
  }
};

Tensor kinematic_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor nodes_b,
    Tensor scans_b,
    Tensor gens_b,
    Tensor kintree) {
  using tmol::utility::connect_backward_pass;

  at::Tensor coords;
  at::Tensor HTs;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(dofs.type(), "forward_kin_op", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result =
                                      ForwardKinDispatch<Dev, Real, Int>::f(
                                          TCAST(dofs),
                                          TCAST(nodes_f),
                                          TCAST(scans_f),
                                          TCAST(gens_f),
                                          TCAST(kintree));

                                  coords = std::get<0>(result).tensor;
                                  HTs = std::get<1>(result).tensor;
                                }));

  return connect_backward_pass({dofs}, coords, [&]() {
    return std::shared_ptr<KinOpBackward>(
        new KinOpBackward(HTs, dofs, nodes_b, scans_b, gens_b, kintree),
        torch::autograd::deleteFunction);
  });
};

Tensor forward_only_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor kintree) {
  using tmol::utility::connect_backward_pass;

  at::Tensor coords;
  // at::Tensor HTs;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(dofs.type(), "forward_kin_only_op", ([&] {
                                  using Real = scalar_t;
                                  constexpr tmol::Device Dev = device_t;

                                  auto result =
                                      ForwardKinDispatch<Dev, Real, Int>::f(
                                          TCAST(dofs),
                                          TCAST(nodes_f),
                                          TCAST(scans_f),
                                          TCAST(gens_f),
                                          TCAST(kintree));

                                  coords = std::get<0>(result).tensor;
                                  // HTs = std::get<1>(result).tensor;
                                }));

  return coords;
};

static auto registry = torch::jit::RegisterOperators()
                           .op("tmol::forward_kin_op", &kinematic_op)
                           .op("tmol::forward_only_kin_op", &forward_only_op);

}  // namespace kinematics
}  // namespace tmol
