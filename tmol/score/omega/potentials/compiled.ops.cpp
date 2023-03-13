#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "params.hh"
#include "dispatch.hh"
#include "omega_pose_score.hh"

namespace tmol {
namespace score {
namespace omega {
namespace potentials {

using namespace tmol::score::common;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class ScoreOp : public torch::autograd::Function<ScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor coords, Tensor params) {
    at::Tensor score;
    at::Tensor dScore;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "omega_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = OmegaDispatch<DispatchMethod, Dev, Real, Int>::f(
              TCAST(coords), TCAST(params));

          score = std::get<0>(result).tensor;
          dScore = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({dScore});

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
    auto dCoords = result[i++];

    return {
        dCoords,
        torch::Tensor(),
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
class OmegaPoseScoreOp
    : public torch::autograd::Function<OmegaPoseScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor block_type_omega_quad_uaids,
      Tensor block_type_atom_downstream_of_conn,
      Tensor global_params) {
    at::Tensor score;
    at::Tensor dscore_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "omega_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              OmegaPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
                  TCAST(coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_block_connections),
                  TCAST(block_type_omega_quad_uaids),
                  TCAST(block_type_atom_downstream_of_conn),
                  TCAST(global_params),
                  coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({dscore_dcoords});
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
    auto dscore_dcoords = result[i++];

    return {
        dscore_dcoords,

        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor omega_pose_scores_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor block_type_omega_quad_uaids,
    Tensor block_type_atom_downstream_of_conn,
    Tensor global_params) {
  return OmegaPoseScoreOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_omega_quad_uaids,
      block_type_atom_downstream_of_conn,
      global_params);
}

template <template <tmol::Device> class DispatchMethod>
Tensor score_op(Tensor coords, Tensor params) {
  return ScoreOp<DispatchMethod>::apply(coords, params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_omega", &score_op<common::ForallDispatch>);
  m.def("omega_pose_scores", &omega_pose_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace omega
}  // namespace score
}  // namespace tmol
