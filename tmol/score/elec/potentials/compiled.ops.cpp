#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "elec_pose_score.hh"

namespace tmol {
namespace score {
namespace elec {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class ElecPoseScoreOp
    : public torch::autograd::Function<ElecPoseScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_min_bond_separation,
      Tensor pose_stack_inter_block_bondsep,

      Tensor block_type_n_atoms,
      Tensor block_type_partial_charge,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,
      Tensor block_type_inter_repr_path_distance,

      Tensor block_type_intra_repr_path_distance,
      Tensor global_params,
      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;
    at::Tensor block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.options(), "elec_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              ElecPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
                  TCAST(coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),

                  TCAST(block_type_n_atoms),
                  TCAST(block_type_partial_charge),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_inter_repr_path_distance),

                  TCAST(block_type_intra_repr_path_distance),
                  TCAST(global_params),
                  output_block_pair_energies,
                  coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          block_neighbors = std::get<2>(result).tensor;
        }));

    if (output_block_pair_energies) {
      // save inputs for deriv call in backwards
      ctx->save_for_backward(
          {coords,
           pose_stack_block_coord_offset,

           pose_stack_block_type,
           pose_stack_min_bond_separation,
           pose_stack_inter_block_bondsep,

           block_type_n_atoms,
           block_type_partial_charge,
           block_type_n_interblock_bonds,
           block_type_atoms_forming_chemical_bonds,
           block_type_inter_repr_path_distance,

           block_type_intra_repr_path_distance,
           global_params,
           block_neighbors});
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
      auto pose_stack_min_bond_separation = saved[i++];
      auto pose_stack_inter_block_bondsep = saved[i++];

      auto block_type_n_atoms = saved[i++];
      auto block_type_partial_charge = saved[i++];
      auto block_type_n_interblock_bonds = saved[i++];
      auto block_type_atoms_forming_chemical_bonds = saved[i++];
      auto block_type_inter_repr_path_distance = saved[i++];

      auto block_type_intra_repr_path_distance = saved[i++];
      auto global_params = saved[i++];
      auto block_neighbors = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          coords.options(), "elec_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = ElecPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    TCAST(coords),
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(pose_stack_block_type),
                    TCAST(pose_stack_min_bond_separation),
                    TCAST(pose_stack_inter_block_bondsep),

                    TCAST(block_type_n_atoms),
                    TCAST(block_type_partial_charge),
                    TCAST(block_type_n_interblock_bonds),
                    TCAST(block_type_atoms_forming_chemical_bonds),
                    TCAST(block_type_inter_repr_path_distance),

                    TCAST(block_type_intra_repr_path_distance),
                    TCAST(global_params),
                    TCAST(block_neighbors),
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
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor elec_pose_scores_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_min_bond_separation,
    Tensor pose_stack_inter_block_bondsep,

    Tensor block_type_n_atoms,
    Tensor block_type_partial_charge,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_inter_repr_path_distance,

    Tensor block_type_intra_repr_path_distance,
    Tensor global_params,
    bool output_block_pair_energies) {
  return ElecPoseScoreOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_min_bond_separation,
      pose_stack_inter_block_bondsep,

      block_type_n_atoms,
      block_type_partial_charge,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_inter_repr_path_distance,

      block_type_intra_repr_path_distance,
      global_params,
      output_block_pair_energies);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("elec_pose_scores", &elec_pose_scores_op<common::DeviceOperations>);
}

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
