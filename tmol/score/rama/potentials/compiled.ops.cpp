#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include <tmol/score/rama/potentials/rama_pose_score.hh>
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

template <template <tmol::Device> class DispatchMethod>
class PoseScoreOp
    : public torch::autograd::Function<PoseScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor block_type_atom_downstream_of_conn,

      Tensor block_type_rama_table,
      Tensor block_type_upper_conn_ind,
      Tensor block_type_is_pro,
      Tensor block_type_rama_torsion_atoms,
      Tensor rama_tables,

      Tensor table_params) {
    at::Tensor score;
    at::Tensor dScore;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "rama_pose_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          // std::cout << "Rama Score Op: " << std::endl;

          auto result =
              RamaPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
                  TCAST(coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_block_connections),
                  TCAST(block_type_atom_downstream_of_conn),
                  TCAST(block_type_rama_table),
                  TCAST(block_type_upper_conn_ind),
                  TCAST(block_type_is_pro),
                  TCAST(block_type_rama_torsion_atoms),
                  TCAST(rama_tables),
                  TCAST(table_params));

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

template <template <tmol::Device> class DispatchMethod>
Tensor pose_score_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor block_type_atom_downstream_of_conn,
    Tensor block_type_rama_table,
    Tensor block_type_upper_conn_ind,
    Tensor block_type_is_pro,
    Tensor block_type_rama_torsion_atoms,
    Tensor rama_tables,
    Tensor table_params) {
  return PoseScoreOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_upper_conn_ind,
      block_type_is_pro,
      block_type_rama_torsion_atoms,
      rama_tables,
      table_params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_rama", &score_op<common::ForallDispatch>);
  m.def("pose_score_rama", &pose_score_op<common::DeviceOperations>);
}

}  // namespace potentials
}  // namespace rama
}  // namespace score
}  // namespace tmol
