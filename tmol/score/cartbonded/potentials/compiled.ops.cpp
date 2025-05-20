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
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_block_connections,
      Tensor atom_paths_from_conn,
      Tensor atom_unique_ids,
      Tensor atom_wildcard_ids,
      Tensor hash_keys,
      Tensor hash_values,
      Tensor cart_subgraphs,
      Tensor cart_subgraph_offsets,
      Tensor max_subgraphs_per_block,
      bool output_block_pair_energies) {
    // Tensor global_paTensor rams) {
    at::Tensor score;
    at::Tensor dscore_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.options(), "cartbonded_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              CartBondedPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  forward(
                      TCAST(coords),
                      TCAST(pose_stack_block_coord_offset),
                      TCAST(pose_stack_block_type),
                      TCAST(pose_stack_inter_block_connections),
                      TCAST(atom_paths_from_conn),
                      TCAST(atom_unique_ids),
                      TCAST(atom_wildcard_ids),
                      TCAST(hash_keys),
                      TCAST(hash_values),
                      TCAST(cart_subgraphs),
                      TCAST(cart_subgraph_offsets),
                      max_subgraphs_per_block.item<int>(),
                      output_block_pair_energies,
                      coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
        }));

    if (output_block_pair_energies) {
      ctx->save_for_backward(
          {coords,
           pose_stack_block_coord_offset,
           pose_stack_block_type,
           pose_stack_inter_block_connections,
           atom_paths_from_conn,
           atom_unique_ids,
           atom_wildcard_ids,
           hash_keys,
           hash_values,
           cart_subgraphs,
           cart_subgraph_offsets,
           max_subgraphs_per_block});
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
      auto atom_paths_from_conn = saved[i++];
      auto atom_unique_ids = saved[i++];
      auto atom_wildcard_ids = saved[i++];
      auto hash_keys = saved[i++];
      auto hash_values = saved[i++];
      auto cart_subgraphs = saved[i++];
      auto cart_subgraph_offsets = saved[i++];
      auto max_subgraphs_per_block = saved[i++];

      // int max_subgraphs_per_block =
      // ctx->saved_data["block_pair_scoring"].toInt();

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          coords.options(), "cartbonded_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result =
                CartBondedPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                    backward(
                        TCAST(coords),
                        TCAST(pose_stack_block_coord_offset),
                        TCAST(pose_stack_block_type),
                        TCAST(pose_stack_inter_block_connections),
                        TCAST(atom_paths_from_conn),
                        TCAST(atom_unique_ids),
                        TCAST(atom_wildcard_ids),
                        TCAST(hash_keys),
                        TCAST(hash_values),
                        TCAST(cart_subgraphs),
                        TCAST(cart_subgraph_offsets),
                        max_subgraphs_per_block.item<int>(),
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
        torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor cartbonded_pose_scores_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_block_connections,
    Tensor atom_paths_from_conn,
    Tensor atom_unique_ids,
    Tensor atom_wildcard_ids,
    Tensor hash_keys,
    Tensor hash_values,
    Tensor cart_subgraphs,
    Tensor cart_subgraph_offsets,
    Tensor max_subgraphs_per_block,
    bool output_block_pair_energies) {
  return CartBondedPoseScoreOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      atom_paths_from_conn,
      atom_unique_ids,
      atom_wildcard_ids,
      hash_keys,
      hash_values,
      cart_subgraphs,
      cart_subgraph_offsets,
      max_subgraphs_per_block,
      output_block_pair_energies);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("cartbonded_pose_scores", &cartbonded_pose_scores_op<DeviceOperations>);
}

// #define PYBIND11_MODULE_(ns, m) PYBIND11_MODULE(ns, m)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cartbonded_pose_scores", &cartbonded_pose_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
