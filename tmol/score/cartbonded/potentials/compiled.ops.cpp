#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/forall_dispatch.hh>

#include <pybind11/pybind11.h>

#include "params.hh"
#include "cartbonded_pose_score.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

using namespace tmol::score::common;

template <template <tmol::Device> class DispatchMethod>
class CartBondedPoseScoreOp
    : public torch::autograd::Function<CartBondedPoseScoreOp<DispatchMethod>> {
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

      Tensor pose_stack_inter_block_connections,
      Tensor atom_paths_from_conn,
      Tensor atom_unique_ids,
      Tensor atom_wildcard_ids,
      Tensor hash_keys,

      Tensor hash_values,
      Tensor cart_subgraphs,
      Tensor cart_subgraph_offsets,
      Tensor cart_subgraph_type_counts,
      Tensor cart_subgraph_type_offsets,

      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;
    at::Tensor dispatch_indices;
    at::Tensor n_output_intxns_for_rot_conn_offset;
    at::Tensor rotconn_for_output_intxn;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "cartbonded_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              CartBondedPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
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

                      TCAST(pose_stack_inter_block_connections),
                      TCAST(atom_paths_from_conn),
                      TCAST(atom_unique_ids),
                      TCAST(atom_wildcard_ids),
                      TCAST(hash_keys),
                      TCAST(hash_values),
                      TCAST(cart_subgraphs),
                      TCAST(cart_subgraph_offsets),
                      TCAST(cart_subgraph_type_counts),
                      TCAST(cart_subgraph_type_offsets),

                      output_block_pair_energies,
                      rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          dispatch_indices = std::get<2>(result).tensor;
          n_output_intxns_for_rot_conn_offset = std::get<3>(result).tensor;
          rotconn_for_output_intxn = std::get<4>(result).tensor;
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

           pose_stack_inter_block_connections,
           atom_paths_from_conn,
           atom_unique_ids,
           atom_wildcard_ids,
           hash_keys,
           hash_values,
           cart_subgraphs,
           cart_subgraph_offsets,
           cart_subgraph_type_counts,
           cart_subgraph_type_offsets,

           dispatch_indices,
           n_output_intxns_for_rot_conn_offset,
           rotconn_for_output_intxn});
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
      // TO DO: make this a function so it's not duplicated everywhere
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
      auto atom_paths_from_conn = saved[i++];
      auto atom_unique_ids = saved[i++];
      auto atom_wildcard_ids = saved[i++];
      auto hash_keys = saved[i++];
      auto hash_values = saved[i++];
      auto cart_subgraphs = saved[i++];
      auto cart_subgraph_offsets = saved[i++];
      auto cart_subgraph_type_counts = saved[i++];
      auto cart_subgraph_type_offsets = saved[i++];

      // Tensors generated during the forward pass
      auto dispatch_indices = saved[i++];
      auto n_output_intxns_for_rot_conn_offset = saved[i++];
      auto rotconn_for_output_intxn = saved[i++];

      // int max_subgraphs_per_block =
      // ctx->saved_data["block_pair_scoring"].toInt();

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "cartbonded_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result =
                CartBondedPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
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

                        TCAST(pose_stack_inter_block_connections),
                        TCAST(atom_paths_from_conn),
                        TCAST(atom_unique_ids),
                        TCAST(atom_wildcard_ids),
                        TCAST(hash_keys),
                        TCAST(hash_values),
                        TCAST(cart_subgraphs),
                        TCAST(cart_subgraph_offsets),
                        TCAST(cart_subgraph_type_counts),
                        TCAST(cart_subgraph_type_offsets),

                        TCAST(dispatch_indices),
                        TCAST(n_output_intxns_for_rot_conn_offset),
                        TCAST(rotconn_for_output_intxn),

                        TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));
    }

    return {
        // Common params
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

        // Cart-bonded specific parameters
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
std::vector<Tensor> cartbonded_pose_scores_op(
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

    Tensor pose_stack_inter_block_connections,
    Tensor atom_paths_from_conn,
    Tensor atom_unique_ids,
    Tensor atom_wildcard_ids,
    Tensor hash_keys,

    Tensor hash_values,
    Tensor cart_subgraphs,
    Tensor cart_subgraph_offsets,
    Tensor cart_subgraph_type_counts,
    Tensor cart_subgraph_type_offsets,

    bool output_block_pair_energies) {
  return CartBondedPoseScoreOp<DispatchMethod>::apply(
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

      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      atom_unique_ids,
      atom_wildcard_ids,
      hash_keys,
      hash_values,
      cart_subgraphs,
      cart_subgraph_offsets,
      cart_subgraph_type_counts,
      cart_subgraph_type_offsets,
      output_block_pair_energies);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("cartbonded_pose_scores", &cartbonded_pose_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
