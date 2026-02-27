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
  static std::vector<Tensor> forward(
      AutogradContext* ctx,

      // common params
      Tensor rot_coords,
      Tensor rot_coord_offset,
      Tensor pose_ind_for_atom,
      Tensor first_rot_for_block,
      Tensor first_rot_block_type,
      Tensor block_ind_for_rot,
      Tensor pose_ind_for_rot,
      Tensor block_type_ind_for_rot,
      Tensor n_rots_for_pose,
      Tensor rot_offset_for_pose,
      Tensor n_rots_for_block,
      Tensor rot_offset_for_block,
      int64_t max_n_rots_per_pose,

      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor block_type_atom_downstream_of_conn,

      Tensor block_type_rama_table,
      Tensor block_type_omega_table,
      Tensor block_type_lower_conn_ind,
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
        rot_coords.options(), "backbone_torsion_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              BackboneTorsionPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  forward(
                      // common params
                      TCAST(rot_coords),
                      TCAST(rot_coord_offset),
                      TCAST(pose_ind_for_atom),
                      TCAST(first_rot_for_block),
                      TCAST(first_rot_block_type),
                      TCAST(block_ind_for_rot),
                      TCAST(pose_ind_for_rot),
                      TCAST(block_type_ind_for_rot),
                      TCAST(n_rots_for_pose),
                      TCAST(rot_offset_for_pose),
                      TCAST(n_rots_for_block),
                      TCAST(rot_offset_for_block),
                      max_n_rots_per_pose,

                      TCAST(pose_stack_block_type),
                      TCAST(pose_stack_inter_block_connections),
                      TCAST(block_type_atom_downstream_of_conn),
                      TCAST(block_type_rama_table),
                      TCAST(block_type_omega_table),
                      TCAST(block_type_lower_conn_ind),
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
      auto max_n_rots_per_pose_tp =
          TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);
      ctx->save_for_backward(
          {rot_coords,
           rot_coord_offset,
           pose_ind_for_atom,
           first_rot_for_block,
           first_rot_block_type,
           block_ind_for_rot,
           pose_ind_for_rot,
           block_type_ind_for_rot,
           n_rots_for_pose,
           rot_offset_for_pose,
           n_rots_for_block,
           rot_offset_for_block,
           max_n_rots_per_pose_tp.tensor,

           pose_stack_block_type,
           pose_stack_inter_block_connections,
           block_type_atom_downstream_of_conn,
           block_type_rama_table,
           block_type_omega_table,
           block_type_lower_conn_ind,
           block_type_upper_conn_ind,
           block_type_is_pro,
           block_type_backbone_torsion_atoms,
           rama_tables,
           rama_table_params,
           omega_tables,
           omega_table_params});
    } else {
      score = score.squeeze(-1).squeeze(-1);
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }

    // Keep the convention that we return two tensors, with the understanding
    // that the second tensor should not be used
    return {score, dscore_dcoords};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_rot_coords;

    // use the number of stashed variables to determine if we are in
    //   block-pair scoring mode or single-score mode
    if (saved.size() == 2) {
      // single-score mode
      auto saved_grads = ctx->get_saved_variables();
      auto saved_grad = saved_grads[0];
      auto pose_ind_for_atom = saved_grads[1];

      tensor_list result;

      auto atom_ingrads = grad_outputs[0].index_select(1, pose_ind_for_atom);

      while (atom_ingrads.dim() < saved_grad.dim()) {
        atom_ingrads = atom_ingrads.unsqueeze(-1);
      }

      result.emplace_back(saved_grad * atom_ingrads);

      int i = 0;

      dV_d_rot_coords = result[i++];

    } else {
      // block-pair mode
      int i = 0;

      // common params
      auto rot_coords = saved[i++];
      auto rot_coord_offset = saved[i++];
      auto pose_ind_for_atom = saved[i++];
      auto first_rot_for_block = saved[i++];
      auto first_rot_block_type = saved[i++];
      auto block_ind_for_rot = saved[i++];
      auto pose_ind_for_rot = saved[i++];
      auto block_type_ind_for_rot = saved[i++];
      auto n_rots_for_pose = saved[i++];
      auto rot_offset_for_pose = saved[i++];
      auto n_rots_for_block = saved[i++];
      auto rot_offset_for_block = saved[i++];
      auto max_n_rots_per_pose =
          TPack<int32_t, 1, tmol::Device::CPU>(saved[i++]).view[0];

      auto pose_stack_block_type = saved[i++];
      auto pose_stack_inter_block_connections = saved[i++];
      auto block_type_atom_downstream_of_conn = saved[i++];
      auto block_type_rama_table = saved[i++];
      auto block_type_omega_table = saved[i++];
      auto block_type_lower_conn_ind = saved[i++];
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
          rot_coords.options(), "backbone_torsion_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = BackboneTorsionPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    // common params
                    TCAST(rot_coords),
                    TCAST(rot_coord_offset),
                    TCAST(pose_ind_for_atom),
                    TCAST(first_rot_for_block),
                    TCAST(first_rot_block_type),
                    TCAST(block_ind_for_rot),
                    TCAST(pose_ind_for_rot),
                    TCAST(block_type_ind_for_rot),
                    TCAST(n_rots_for_pose),
                    TCAST(rot_offset_for_pose),
                    TCAST(n_rots_for_block),
                    TCAST(rot_offset_for_block),
                    max_n_rots_per_pose,

                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_block_connections),
                    TCAST(block_type_atom_downstream_of_conn),
                    TCAST(block_type_rama_table),
                    TCAST(block_type_omega_table),
                    TCAST(block_type_lower_conn_ind),
                    TCAST(block_type_upper_conn_ind),
                    TCAST(block_type_is_pro),
                    TCAST(block_type_backbone_torsion_atoms),
                    TCAST(rama_tables),
                    TCAST(rama_table_params),
                    TCAST(omega_tables),
                    TCAST(omega_table_params),
                    TCAST(dTdV));

            dV_d_rot_coords = result.tensor;
          }));
    }

    return {
        // common: rot_coords + 12 more arguments
        dV_d_rot_coords,

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

        // 14 more arguments
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
class BackboneTorsionRotamerScoreOp
    : public torch::autograd::Function<
          BackboneTorsionRotamerScoreOp<DispatchMethod>> {
 public:
  static std::vector<Tensor> forward(
      AutogradContext* ctx,

      // common params
      Tensor rot_coords,
      Tensor rot_coord_offset,
      Tensor pose_ind_for_atom,
      Tensor first_rot_for_block,
      Tensor first_rot_block_type,
      Tensor block_ind_for_rot,
      Tensor pose_ind_for_rot,
      Tensor block_type_ind_for_rot,
      Tensor n_rots_for_pose,
      Tensor rot_offset_for_pose,
      Tensor n_rots_for_block,
      Tensor rot_offset_for_block,
      int64_t max_n_rots_per_pose,

      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor block_type_atom_downstream_of_conn,

      Tensor block_type_rama_table,
      Tensor block_type_omega_table,
      Tensor block_type_lower_conn_ind,
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
    at::Tensor dispatch_indices;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "backbone_torsion_rotamer_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = BackboneTorsionRotamerScoreDispatch<
              DispatchMethod,
              Dev,
              Real,
              Int>::
              forward(
                  // common params
                  TCAST(rot_coords),
                  TCAST(rot_coord_offset),
                  TCAST(pose_ind_for_atom),
                  TCAST(first_rot_for_block),
                  TCAST(first_rot_block_type),
                  TCAST(block_ind_for_rot),
                  TCAST(pose_ind_for_rot),
                  TCAST(block_type_ind_for_rot),
                  TCAST(n_rots_for_pose),
                  TCAST(rot_offset_for_pose),
                  TCAST(n_rots_for_block),
                  TCAST(rot_offset_for_block),
                  max_n_rots_per_pose,

                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_block_connections),
                  TCAST(block_type_atom_downstream_of_conn),
                  TCAST(block_type_rama_table),
                  TCAST(block_type_omega_table),
                  TCAST(block_type_lower_conn_ind),
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
          dispatch_indices = std::get<2>(result).tensor;
        }));

    if (output_block_pair_energies) {
      auto max_n_rots_per_pose_tp =
          TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);
      ctx->save_for_backward(
          {rot_coords,
           rot_coord_offset,
           pose_ind_for_atom,
           first_rot_for_block,
           first_rot_block_type,
           block_ind_for_rot,
           pose_ind_for_rot,
           block_type_ind_for_rot,
           n_rots_for_pose,
           rot_offset_for_pose,
           n_rots_for_block,
           rot_offset_for_block,
           max_n_rots_per_pose_tp.tensor,

           pose_stack_block_type,
           pose_stack_inter_block_connections,
           block_type_atom_downstream_of_conn,
           block_type_rama_table,
           block_type_omega_table,
           block_type_lower_conn_ind,
           block_type_upper_conn_ind,
           block_type_is_pro,
           block_type_backbone_torsion_atoms,
           rama_tables,
           rama_table_params,
           omega_tables,
           omega_table_params,
           dispatch_indices});
    } else {
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }

    return {score, dispatch_indices};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_rot_coords;

    // use the number of stashed variables to determine if we are in
    //   block-pair scoring mode or single-score mode
    if (saved.size() == 2) {
      // single-score mode
      auto saved_grads = ctx->get_saved_variables();
      auto saved_grad = saved_grads[0];
      auto pose_ind_for_atom = saved_grads[1];

      tensor_list result;

      auto atom_ingrads = grad_outputs[0].index_select(1, pose_ind_for_atom);

      while (atom_ingrads.dim() < saved_grad.dim()) {
        atom_ingrads = atom_ingrads.unsqueeze(-1);
      }

      result.emplace_back(saved_grad * atom_ingrads);

      int i = 0;

      dV_d_rot_coords = result[i++];

    } else {
      // block-pair mode
      int i = 0;

      // common params
      auto rot_coords = saved[i++];
      auto rot_coord_offset = saved[i++];
      auto pose_ind_for_atom = saved[i++];
      auto first_rot_for_block = saved[i++];
      auto first_rot_block_type = saved[i++];
      auto block_ind_for_rot = saved[i++];
      auto pose_ind_for_rot = saved[i++];
      auto block_type_ind_for_rot = saved[i++];
      auto n_rots_for_pose = saved[i++];
      auto rot_offset_for_pose = saved[i++];
      auto n_rots_for_block = saved[i++];
      auto rot_offset_for_block = saved[i++];
      auto max_n_rots_per_pose =
          TPack<int32_t, 1, tmol::Device::CPU>(saved[i++]).view[0];

      auto pose_stack_block_type = saved[i++];
      auto pose_stack_inter_block_connections = saved[i++];
      auto block_type_atom_downstream_of_conn = saved[i++];
      auto block_type_rama_table = saved[i++];
      auto block_type_omega_table = saved[i++];
      auto block_type_lower_conn_ind = saved[i++];
      auto block_type_upper_conn_ind = saved[i++];
      auto block_type_is_pro = saved[i++];
      auto block_type_backbone_torsion_atoms = saved[i++];
      auto rama_tables = saved[i++];
      auto rama_table_params = saved[i++];
      auto omega_tables = saved[i++];
      auto omega_table_params = saved[i++];
      auto dispatch_indices = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(),
          "backbone_torsion_rotamer_score_backward",
          ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = BackboneTorsionRotamerScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    // common params
                    TCAST(rot_coords),
                    TCAST(rot_coord_offset),
                    TCAST(pose_ind_for_atom),
                    TCAST(first_rot_for_block),
                    TCAST(first_rot_block_type),
                    TCAST(block_ind_for_rot),
                    TCAST(pose_ind_for_rot),
                    TCAST(block_type_ind_for_rot),
                    TCAST(n_rots_for_pose),
                    TCAST(rot_offset_for_pose),
                    TCAST(n_rots_for_block),
                    TCAST(rot_offset_for_block),
                    max_n_rots_per_pose,

                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_inter_block_connections),
                    TCAST(block_type_atom_downstream_of_conn),
                    TCAST(block_type_rama_table),
                    TCAST(block_type_omega_table),
                    TCAST(block_type_lower_conn_ind),
                    TCAST(block_type_upper_conn_ind),
                    TCAST(block_type_is_pro),
                    TCAST(block_type_backbone_torsion_atoms),
                    TCAST(rama_tables),
                    TCAST(rama_table_params),
                    TCAST(omega_tables),
                    TCAST(omega_table_params),
                    TCAST(dispatch_indices),
                    TCAST(dTdV));

            dV_d_rot_coords = result.tensor;
          }));
    }

    return {
        // common: rot_coords + 12 more arguments
        dV_d_rot_coords,

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

        // 14 more arguments
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
std::vector<Tensor> backbone_torsion_pose_score_op(
    // common params
    Tensor rot_coords,
    Tensor rot_coord_offset,
    Tensor pose_ind_for_atom,
    Tensor first_rot_for_block,
    Tensor first_rot_block_type,
    Tensor block_ind_for_rot,
    Tensor pose_ind_for_rot,
    Tensor block_type_ind_for_rot,
    Tensor n_rots_for_pose,
    Tensor rot_offset_for_pose,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    int64_t max_n_rots_per_pose,

    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor block_type_atom_downstream_of_conn,

    Tensor block_type_rama_table,
    Tensor block_type_omega_table,
    Tensor block_type_lower_conn_ind,
    Tensor block_type_upper_conn_ind,
    Tensor block_type_is_pro,
    Tensor block_type_backbone_torsion_atoms,
    Tensor rama_tables,
    Tensor rama_table_params,
    Tensor omega_tables,
    Tensor omega_table_params,
    bool output_block_pair_energies) {
  return BackboneTorsionPoseScoreOp<DispatchMethod>::apply(
      // common params
      rot_coords,
      rot_coord_offset,
      pose_ind_for_atom,
      first_rot_for_block,
      first_rot_block_type,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
      n_rots_for_pose,
      rot_offset_for_pose,
      n_rots_for_block,
      rot_offset_for_block,
      max_n_rots_per_pose,

      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_omega_table,
      block_type_lower_conn_ind,
      block_type_upper_conn_ind,
      block_type_is_pro,
      block_type_backbone_torsion_atoms,
      rama_tables,
      rama_table_params,
      omega_tables,
      omega_table_params,
      output_block_pair_energies);
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> backbone_torsion_rotamer_score_op(
    // common params
    Tensor rot_coords,
    Tensor rot_coord_offset,
    Tensor pose_ind_for_atom,
    Tensor first_rot_for_block,
    Tensor first_rot_block_type,
    Tensor block_ind_for_rot,
    Tensor pose_ind_for_rot,
    Tensor block_type_ind_for_rot,
    Tensor n_rots_for_pose,
    Tensor rot_offset_for_pose,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    int64_t max_n_rots_per_pose,

    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor block_type_atom_downstream_of_conn,

    Tensor block_type_rama_table,
    Tensor block_type_omega_table,
    Tensor block_type_lower_conn_ind,
    Tensor block_type_upper_conn_ind,
    Tensor block_type_is_pro,
    Tensor block_type_backbone_torsion_atoms,
    Tensor rama_tables,
    Tensor rama_table_params,
    Tensor omega_tables,
    Tensor omega_table_params,
    bool output_block_pair_energies) {
  return BackboneTorsionRotamerScoreOp<DispatchMethod>::apply(
      // common params
      rot_coords,
      rot_coord_offset,
      pose_ind_for_atom,
      first_rot_for_block,
      first_rot_block_type,
      block_ind_for_rot,
      pose_ind_for_rot,
      block_type_ind_for_rot,
      n_rots_for_pose,
      rot_offset_for_pose,
      n_rots_for_block,
      rot_offset_for_block,
      max_n_rots_per_pose,

      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn,
      block_type_rama_table,
      block_type_omega_table,
      block_type_lower_conn_ind,
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
  m.def(
      "backbone_torsion_rotamer_score",
      &backbone_torsion_rotamer_score_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace backbone_torsion
}  // namespace score
}  // namespace tmol
