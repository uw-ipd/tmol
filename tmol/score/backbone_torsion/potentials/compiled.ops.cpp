#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "params.hh"
#include "backbone_torsion_pose_score.hh"

namespace tmol {
namespace score {
namespace backbone_torsion {
namespace potentials {

using namespace tmol::score::common;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class BackboneTorsionPoseScoreOp
    : public torch::autograd::Function<
          BackboneTorsionPoseScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor block_type_atom_downstream_of_conn,

      Tensor block_type_rama_table,
      Tensor block_type_omega_table,
      Tensor block_type_upper_conn_ind,
      Tensor block_type_is_pro,
      Tensor block_type_backbone_torsion_atoms,
      Tensor rama_tables,
      Tensor rama_table_params,
      Tensor omega_tables,
      Tensor omega_table_params,
      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.options(), "omega_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              BackboneTorsionPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  forward(
                      TCAST(coords),
                      TCAST(pose_stack_block_coord_offset),
                      TCAST(pose_stack_block_type),
                      TCAST(pose_stack_inter_block_connections),
                      TCAST(block_type_atom_downstream_of_conn),
                      TCAST(block_type_rama_table),
                      TCAST(block_type_omega_table),
                      TCAST(block_type_upper_conn_ind),
                      TCAST(block_type_is_pro),
                      TCAST(block_type_backbone_torsion_atoms),
                      TCAST(rama_tables),
                      TCAST(rama_table_params),
                      TCAST(omega_tables),
                      TCAST(omega_table_params),
                      output_block_pair_energies);

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
        }));

    if (output_block_pair_energies) {
      ctx->save_for_backward(
          {coords,
           pose_stack_block_coord_offset,
           pose_stack_block_type,
           pose_stack_inter_block_connections,
           block_type_atom_downstream_of_conn,
           block_type_rama_table,
           block_type_omega_table,
           block_type_upper_conn_ind,
           block_type_is_pro,
           block_type_backbone_torsion_atoms,
           rama_tables,
           rama_table_params,
           omega_tables,
           omega_table_params});
    } else {
      score = score.squeeze(-1).squeeze(-1);  // remove final 2 "dummy" dims
      ctx->save_for_backward({dscore_dcoords});
    }
    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_pose_coords;

    // use the number of stashed variables to determine if we are in
    //   block-pair scoring mode or single-score mode
    if (saved.size() == 1) {
      // single-score mode
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
      dV_d_pose_coords = result[i++];

    } else {
      // block-pair mode
      int i = 0;

      auto coords = saved[i++];
      auto pose_stack_block_coord_offset = saved[i++];
      auto pose_stack_block_type = saved[i++];
      auto pose_stack_inter_block_connections = saved[i++];
      auto block_type_atom_downstream_of_conn = saved[i++];
      auto block_type_rama_table = saved[i++];
      auto block_type_omega_table = saved[i++];
      auto block_type_upper_conn_ind = saved[i++];
      auto block_type_is_pro = saved[i++];
      auto block_type_backbone_torsion_atoms = saved[i++];
      auto rama_tables = saved[i++];
      auto rama_table_params = saved[i++];
      auto omega_tables = saved[i++];
      auto omega_table_params = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          coords.options(), "disulfide_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = BackboneTorsionPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    TCAST(coords),
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_block_connections),
                    TCAST(block_type_atom_downstream_of_conn),
                    TCAST(block_type_rama_table),
                    TCAST(block_type_omega_table),
                    TCAST(block_type_upper_conn_ind),
                    TCAST(block_type_is_pro),
                    TCAST(block_type_backbone_torsion_atoms),
                    TCAST(rama_tables),
                    TCAST(rama_table_params),
                    TCAST(omega_tables),
                    TCAST(omega_table_params),
                    TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));
    }

    return {
        dV_d_pose_coords,

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

        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor backbone_torsion_pose_score_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor block_type_atom_downstream_of_conn,

    Tensor block_type_rama_table,
    Tensor block_type_omega_table,
    Tensor block_type_upper_conn_ind,
    Tensor block_type_is_pro,
    Tensor block_type_backbone_torsion_atoms,
    Tensor rama_tables,
    Tensor rama_table_params,
    Tensor omega_tables,
    Tensor omega_table_params,
    bool output_block_pair_energies) {
  return BackboneTorsionPoseScoreOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_omega_table,
      block_type_upper_conn_ind,
      block_type_is_pro,
      block_type_backbone_torsion_atoms,
      rama_tables,
      rama_table_params,
      omega_tables,
      omega_table_params,
      output_block_pair_energies);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def(
      "backbone_torsion_pose_score",
      &backbone_torsion_pose_score_op<DeviceOperations>);
}

// #define PYBIND11_MODULE_(ns, m) PYBIND11_MODULE(ns, m)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "backbone_torsion_pose_score",
      &backbone_torsion_pose_score_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
