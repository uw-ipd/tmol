#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/whole_pose_scoring.hh>

#include "params.hh"
#include "metal_coordination_pose_score.hh"

namespace tmol {
namespace score {
namespace metal {
namespace potentials {

ContextManager mgr;

using namespace tmol::score::common;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class MetalCoordinationPoseScoreOp
    : public torch::autograd::Function<
          MetalCoordinationPoseScoreOp<DispatchMethod>> {
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

      // custom params
      Tensor pose_stack_inter_block_connections,
      Tensor conn_atom,
      Tensor conn_metal,
      Tensor conn_virt,
      Tensor conn_key,
      Tensor site_params,
      Tensor fan_atoms,
      Tensor fan_params,
      Tensor bridge_internal_metal,
      Tensor bridge_internal_d0,
      Tensor bridge_params,

      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "metal_coordination_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = MetalCoordinationPoseScoreDispatch<
              DispatchMethod,
              Dev,
              Real,
              Int>::
              forward(
                  mgr,
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

                  TCAST(pose_stack_inter_block_connections),
                  TCAST(conn_atom),
                  TCAST(conn_metal),
                  TCAST(conn_virt),
                  TCAST(conn_key),
                  TCAST(site_params),
                  TCAST(fan_atoms),
                  TCAST(fan_params),
                  TCAST(bridge_internal_metal),
                  TCAST(bridge_internal_d0),
                  TCAST(bridge_params),
                  output_block_pair_energies,
                  rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
        }));

    if (output_block_pair_energies) {
      // save inputs for deriv call in backwards
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

           pose_stack_inter_block_connections,
           conn_atom,
           conn_metal,
           conn_virt,
           conn_key,
           site_params,
           fan_atoms,
           fan_params,
           bridge_internal_metal,
           bridge_internal_d0,
           bridge_params});
    } else {
      score = score.squeeze(-1).squeeze(-1);
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }
    // Stick to convention: return two tensors, with the understanding
    // that no one will use the second tensor
    return {score, dscore_dcoords};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_pose_coords;

    // use the number of stashed variables to determine if we are in
    //   block-pair scoring mode or single-score mode
    if (saved.size() == 2) {
      // single-score mode
      auto saved_grad = saved[0];

      tensor_list result;
      result.emplace_back(
          common::accumulate_whole_pose_gradients(saved_grad, grad_outputs[0]));

      int i = 0;
      dV_d_pose_coords = result[i++];

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

      auto pose_stack_inter_block_connections = saved[i++];
      auto conn_atom = saved[i++];
      auto conn_metal = saved[i++];
      auto conn_virt = saved[i++];
      auto conn_key = saved[i++];
      auto site_params = saved[i++];
      auto fan_atoms = saved[i++];
      auto fan_params = saved[i++];
      auto bridge_internal_metal = saved[i++];
      auto bridge_internal_d0 = saved[i++];
      auto bridge_params = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "metal_coordination_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = MetalCoordinationPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    mgr,
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

                    TCAST(pose_stack_inter_block_connections),
                    TCAST(conn_atom),
                    TCAST(conn_metal),
                    TCAST(conn_virt),
                    TCAST(conn_key),
                    TCAST(site_params),
                    TCAST(fan_atoms),
                    TCAST(fan_params),
                    TCAST(bridge_internal_metal),
                    TCAST(bridge_internal_d0),
                    TCAST(bridge_params),
                    TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));
    }

    return {
        // 13 common params including dV_d_pose_coords
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

        // 12 custom params
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
class MetalCoordinationRotamerScoreOp
    : public torch::autograd::Function<
          MetalCoordinationRotamerScoreOp<DispatchMethod>> {
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

      // custom params
      Tensor pose_stack_inter_block_connections,
      Tensor conn_atom,
      Tensor conn_metal,
      Tensor conn_virt,
      Tensor conn_key,
      Tensor site_params,
      Tensor fan_atoms,
      Tensor fan_params,
      Tensor bridge_internal_metal,
      Tensor bridge_internal_d0,
      Tensor bridge_params,

      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;
    at::Tensor dispatch_indices;
    at::Tensor terms_for_dispatch;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "metal_coordination_rotamer_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = MetalCoordinationRotamerScoreDispatch<
              DispatchMethod,
              Dev,
              Real,
              Int>::
              forward(
                  mgr,
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

                  TCAST(pose_stack_inter_block_connections),
                  TCAST(conn_atom),
                  TCAST(conn_metal),
                  TCAST(conn_virt),
                  TCAST(conn_key),
                  TCAST(site_params),
                  TCAST(fan_atoms),
                  TCAST(fan_params),
                  TCAST(bridge_internal_metal),
                  TCAST(bridge_internal_d0),
                  TCAST(bridge_params),
                  output_block_pair_energies,
                  rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          dispatch_indices = std::get<2>(result).tensor;
          terms_for_dispatch = std::get<3>(result).tensor;
        }));

    if (output_block_pair_energies) {
      // save inputs for deriv call in backwards
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

           pose_stack_inter_block_connections,
           conn_atom,
           conn_metal,
           conn_virt,
           conn_key,
           site_params,
           fan_atoms,
           fan_params,
           bridge_internal_metal,
           bridge_internal_d0,
           bridge_params,
           terms_for_dispatch});
    } else {
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }
    return {score, dispatch_indices};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_pose_coords;

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
      dV_d_pose_coords = result[i++];

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

      auto pose_stack_inter_block_connections = saved[i++];
      auto conn_atom = saved[i++];
      auto conn_metal = saved[i++];
      auto conn_virt = saved[i++];
      auto conn_key = saved[i++];
      auto site_params = saved[i++];
      auto fan_atoms = saved[i++];
      auto fan_params = saved[i++];
      auto bridge_internal_metal = saved[i++];
      auto bridge_internal_d0 = saved[i++];
      auto bridge_params = saved[i++];
      auto terms_for_dispatch = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(),
          "metal_coordination_rotamer_score_backward",
          ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = MetalCoordinationRotamerScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    mgr,
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

                    TCAST(pose_stack_inter_block_connections),
                    TCAST(conn_atom),
                    TCAST(conn_metal),
                    TCAST(conn_virt),
                    TCAST(conn_key),
                    TCAST(site_params),
                    TCAST(fan_atoms),
                    TCAST(fan_params),
                    TCAST(bridge_internal_metal),
                    TCAST(bridge_internal_d0),
                    TCAST(bridge_params),
                    TCAST(terms_for_dispatch),
                    TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));
    }

    return {
        // 13 common params including dV_d_pose_coords
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

        // 12 custom params
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
std::vector<Tensor> metal_coordination_pose_scores_op(
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

    // custom params
    Tensor pose_stack_inter_block_connections,
    Tensor conn_atom,
    Tensor conn_metal,
    Tensor conn_virt,
    Tensor conn_key,
    Tensor site_params,
    Tensor fan_atoms,
    Tensor fan_params,
    Tensor bridge_internal_metal,
    Tensor bridge_internal_d0,
    Tensor bridge_params,

    bool output_block_pair_energies) {
  return MetalCoordinationPoseScoreOp<DispatchMethod>::apply(
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

      // custom params
      pose_stack_inter_block_connections,
      conn_atom,
      conn_metal,
      conn_virt,
      conn_key,
      site_params,
      fan_atoms,
      fan_params,
      bridge_internal_metal,
      bridge_internal_d0,
      bridge_params,

      output_block_pair_energies);
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> metal_coordination_rotamer_scores_op(
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
    // only the pair terms enumerate rotamer pairs
    Tensor /*lockstep_group_for_block*/,
    int64_t max_n_rots_per_pose,

    // custom params
    Tensor pose_stack_inter_block_connections,
    Tensor conn_atom,
    Tensor conn_metal,
    Tensor conn_virt,
    Tensor conn_key,
    Tensor site_params,
    Tensor fan_atoms,
    Tensor fan_params,
    Tensor bridge_internal_metal,
    Tensor bridge_internal_d0,
    Tensor bridge_params,

    bool output_block_pair_energies) {
  return MetalCoordinationRotamerScoreOp<DispatchMethod>::apply(
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

      // custom params
      pose_stack_inter_block_connections,
      conn_atom,
      conn_metal,
      conn_virt,
      conn_key,
      site_params,
      fan_atoms,
      fan_params,
      bridge_internal_metal,
      bridge_internal_d0,
      bridge_params,

      output_block_pair_energies);
}
// See https://stackoverflow.com/a/3221914
TORCH_LIBRARY(tmol_metal, m) {
  m.def(
      "metal_coordination_pose_scores",
      &metal_coordination_pose_scores_op<DeviceOperations>);
  m.def(
      "metal_coordination_rotamer_scores",
      &metal_coordination_rotamer_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace metal
}  // namespace score
}  // namespace tmol
