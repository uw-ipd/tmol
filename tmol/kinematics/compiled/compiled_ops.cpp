#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>

#include "common.hh"
#include "common_dispatch.hh"
#include "params.hh"

namespace tmol {
namespace kinematics {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class KinematicOp : public torch::autograd::Function<KinematicOp> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor dofs,
      Tensor nodes_f,
      Tensor scans_f,
      Tensor gens_f,
      Tensor nodes_b,
      Tensor scans_b,
      Tensor gens_b,
      Tensor kintree) {
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

    ctx->save_for_backward({HTs, dofs, nodes_b, scans_b, gens_b, kintree});

    return coords;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    int i = 0;
    auto HTs = saved[i++];
    auto dofs = saved[i++];
    auto nodes_b = saved[i++];
    auto scans_b = saved[i++];
    auto gens_b = saved[i++];
    auto kintree = saved[i++];

    at::Tensor dV_ddof;
    using Int = int32_t;
    auto dVdx = grad_outputs[0];
    TMOL_DISPATCH_FLOATING_DEVICE(HTs.type(), "kin_deriv_op", ([&] {
                                    using Real = scalar_t;
                                    constexpr tmol::Device Dev = device_t;

                                    auto result =
                                        KinDerivDispatch<Dev, Real, Int>::f(
                                            TCAST(dVdx),
                                            TCAST(HTs),
                                            TCAST(dofs),
                                            TCAST(nodes_b),
                                            TCAST(scans_b),
                                            TCAST(gens_b),
                                            TCAST(kintree));

                                    dV_ddof = result.tensor;
                                  }));

    return {
        dV_ddof,
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

Tensor kinematic_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor nodes_b,
    Tensor scans_b,
    Tensor gens_b,
    Tensor kintree) {
  return KinematicOp::apply(
      dofs, nodes_f, scans_f, gens_f, nodes_b, scans_b, gens_b, kintree);
}

Tensor forward_only_op(
    Tensor dofs,
    Tensor nodes_f,
    Tensor scans_f,
    Tensor gens_f,
    Tensor kintree
) {

  at::Tensor coords;

  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      dofs.type(), "forward_kin_only_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = ForwardKinDispatch<Dev, Real, Int>::f(
            TCAST(dofs),
            TCAST(nodes_f),
            TCAST(scans_f),
            TCAST(gens_f),
            TCAST(kintree));

        coords = std::get<0>(result).tensor;
      }));

  return coords;

};


// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)

TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("forward_kin_op", &kinematic_op);
  m.def("forward_only_op", &forward_only_op);
}

}  // namespace kinematics
}  // namespace tmol
