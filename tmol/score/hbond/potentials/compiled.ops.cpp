#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include <tmol/score/hbond/potentials/dispatch.hh>
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
      Tensor global_params,
      bool output_block_pair_energies

  ) {
    at::Tensor score;
    at::Tensor dscore_dcoords;
    at::Tensor block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "hbond_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              HBondPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
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
      auto pose_stack_inter_residue_connections = saved[i++];
      auto pose_stack_min_bond_separation = saved[i++];

      auto pose_stack_inter_block_bondsep = saved[i++];
      auto block_type_n_atoms = saved[i++];
      auto block_type_n_interblock_bonds = saved[i++];
      auto block_type_atoms_forming_chemical_bonds = saved[i++];
      auto block_type_n_all_bonds = saved[i++];

      auto block_type_all_bonds = saved[i++];
      auto block_type_atom_all_bond_ranges = saved[i++];
      auto block_type_tile_n_donH = saved[i++];
      auto block_type_tile_n_acc = saved[i++];
      auto block_type_tile_donH_inds = saved[i++];

      auto block_type_tile_acc_inds = saved[i++];
      auto block_type_tile_donor_type = saved[i++];
      auto block_type_tile_acceptor_type = saved[i++];
      auto block_type_tile_hybridization = saved[i++];
      auto block_type_atom_is_hydrogen = saved[i++];

      auto block_type_path_distance = saved[i++];
      auto pair_params = saved[i++];
      auto pair_polynomials = saved[i++];
      auto global_params = saved[i++];
      auto block_neighbors = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          coords.type(), "hbond_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = HBondPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
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
                    TCAST(block_neighbors),
                    TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));
    }

    return {dV_d_pose_coords, torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor()};
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
    Tensor global_params,
    bool output_block_pair_energies) {
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
      global_params,
      output_block_pair_energies);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_hbond", &score_op<HBondDispatch, common::AABBDispatch>);
  m.def("hbond_pose_scores", &hbond_pose_scores_op<common::DeviceOperations>);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
