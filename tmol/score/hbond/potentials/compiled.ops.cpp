#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include <tmol/score/hbond/potentials/hbond_pose_score.hh>

#include <tmol/utility/nvtx.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class HBondPoseScoresOp
    : public torch::autograd::Function<HBondPoseScoresOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
      Tensor pose_stack_inter_residue_connections,
      Tensor pose_stack_min_bond_separation,

      Tensor pose_stack_inter_block_bondsep,
      Tensor block_type_n_atoms,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,
      Tensor block_type_n_all_bonds,

      Tensor block_type_all_bonds,
      Tensor block_type_atom_all_bond_ranges,
      Tensor block_type_tile_n_donH,
      Tensor block_type_tile_n_acc,
      Tensor block_type_tile_donH_inds,

      Tensor block_type_tile_acc_inds,
      Tensor block_type_tile_donor_type,
      Tensor block_type_tile_acceptor_type,
      Tensor block_type_tile_hybridization,
      Tensor block_type_atom_is_hydrogen,

      Tensor block_type_path_distance,
      Tensor pair_params,
      Tensor pair_polynomials,
      Tensor global_params

  ) {
    at::Tensor score;
    at::Tensor dscore_dcoords;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "hbond_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              HBondPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
                  TCAST(coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_inter_residue_connections),
                  TCAST(pose_stack_min_bond_separation),

                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_n_all_bonds),

                  TCAST(block_type_all_bonds),
                  TCAST(block_type_atom_all_bond_ranges),
                  TCAST(block_type_tile_n_donH),
                  TCAST(block_type_tile_n_acc),
                  TCAST(block_type_tile_donH_inds),

                  TCAST(block_type_tile_acc_inds),
                  TCAST(block_type_tile_donor_type),
                  TCAST(block_type_tile_acceptor_type),
                  TCAST(block_type_tile_hybridization),
                  TCAST(block_type_atom_is_hydrogen),

                  TCAST(block_type_path_distance),
                  TCAST(pair_params),
                  TCAST(pair_polynomials),
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
        dscore_dcoords,  torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor hbond_pose_scores_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
    Tensor pose_stack_inter_residue_connections,
    Tensor pose_stack_min_bond_separation,

    Tensor pose_stack_inter_block_bondsep,
    Tensor block_type_n_atoms,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_n_all_bonds,

    Tensor block_type_all_bonds,
    Tensor block_type_atom_all_bond_ranges,
    Tensor block_type_tile_n_donH,
    Tensor block_type_tile_n_acc,
    Tensor block_type_tile_donH_inds,

    Tensor block_type_tile_acc_inds,
    Tensor block_type_tile_donor_type,
    Tensor block_type_tile_acceptor_type,
    Tensor block_type_tile_hybridization,
    Tensor block_type_atom_is_hydrogen,

    Tensor block_type_path_distance,
    Tensor pair_params,
    Tensor pair_polynomials,
    Tensor global_params) {
  return HBondPoseScoresOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_residue_connections,
      pose_stack_min_bond_separation,

      pose_stack_inter_block_bondsep,
      block_type_n_atoms,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_n_all_bonds,

      block_type_all_bonds,
      block_type_atom_all_bond_ranges,
      block_type_tile_n_donH,
      block_type_tile_n_acc,
      block_type_tile_donH_inds,

      block_type_tile_acc_inds,
      block_type_tile_donor_type,
      block_type_tile_acceptor_type,
      block_type_tile_hybridization,
      block_type_atom_is_hydrogen,

      block_type_path_distance,
      pair_params,
      pair_polynomials,
      global_params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("hbond_pose_scores", &hbond_pose_scores_op<common::DeviceOperations>);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
