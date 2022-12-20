#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include "dispatch.hh"
#include <tmol/utility/nvtx.hh>

namespace tmol {
namespace score {
namespace hbond {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
class ScoreOp
    : public torch::autograd::Function<ScoreOp<ScoreDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor donor_coords,
      Tensor acceptor_coords,
      Tensor Dinds,
      Tensor H,
      Tensor donor_type,
      Tensor A,
      Tensor B,
      Tensor B0,
      Tensor acceptor_type,
      Tensor pair_params,
      Tensor pair_polynomials,
      Tensor global_params) {
    at::Tensor score;
    at::Tensor dV_d_don;
    at::Tensor dV_d_acc;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        donor_coords.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = HBondDispatch<DispatchMethod, Dev, Real, Int>::f(
              TCAST(donor_coords),
              TCAST(acceptor_coords),
              TCAST(Dinds),
              TCAST(H),
              TCAST(donor_type),
              TCAST(A),
              TCAST(B),
              TCAST(B0),
              TCAST(acceptor_type),
              TCAST(pair_params),
              TCAST(pair_polynomials),
              TCAST(global_params));

          score = std::get<0>(result).tensor;
          dV_d_don = std::get<1>(result).tensor;
          dV_d_acc = std::get<2>(result).tensor;
        }));

    ctx->save_for_backward({dV_d_don, dV_d_acc});

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
    auto dT_d_don = result[i++];
    auto dT_d_acc = result[i++];

    return {
        dT_d_don,
        dT_d_acc,
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

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
Tensor score_op(
    Tensor donor_coords,
    Tensor acceptor_coords,
    Tensor Dinds,
    Tensor H,
    Tensor donor_type,
    Tensor A,
    Tensor B,
    Tensor B0,
    Tensor acceptor_type,
    Tensor pair_params,
    Tensor pair_polynomials,
    Tensor global_params) {
  return ScoreOp<ScoreDispatch, DispatchMethod>::apply(
      donor_coords,
      acceptor_coords,
      Dinds,
      H,
      donor_type,
      A,
      B,
      B0,
      acceptor_type,
      pair_params,
      pair_polynomials,
      global_params);
}

template <template <tmol::Device> class DispatchMethod>
class HBondPoseScoresOp
    : public torch::autograd::Function<HBondPoseScoresOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor posck_stack_block_coord_offset,

      Tensor pose_stack_block_type,
      Tensor pose_stack_min_bond_separation,
      Tensor pose_stack_inter_block_bondsep,
      Tensor block_type_n_atoms,
      Tensor block_type_n_heavy_atoms_in_tile,

      Tensor block_type_heavy_atoms_in_tile,
      Tensor block_type_atom_types,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,
      Tensor block_type_path_distance,

      Tensor type_params,
      Tensor global_params) {
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
                  TCAST(posck_stack_block_coord_offset),

                  TCAST(pose_stack_block_type),
                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_heavy_atoms_in_tile),

                  TCAST(block_type_heavy_atoms_in_tile),
                  TCAST(block_type_atom_types),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_path_distance),

                  TCAST(type_params),
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
Tensor ljlk_pose_scores_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,

    Tensor pose_stack_block_type,
    Tensor pose_stack_min_bond_separation,
    Tensor pose_stack_inter_block_bondsep,
    Tensor block_type_n_atoms,
    Tensor block_type_n_heavy_atoms_in_tile,

    Tensor block_type_heavy_atoms_in_tile,
    Tensor block_type_atom_types,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_path_distance,

    Tensor ljlk_type_params,
    Tensor global_params) {
  return HBondPoseScoresOp<DispatchMethod>::apply(
      coords,
      pose_stack_block_coord_offset,

      pose_stack_block_type,
      pose_stack_min_bond_separation,
      pose_stack_inter_block_bondsep,
      block_type_n_atoms,
      block_type_n_heavy_atoms_in_tile,

      block_type_heavy_atoms_in_tile,
      block_type_atom_types,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_path_distance,

      ljlk_type_params,
      global_params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_hbond", &score_op<HBondDispatch, common::AABBDispatch>);
  m.def("hbond_pose_scores", &hbond_pose_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
