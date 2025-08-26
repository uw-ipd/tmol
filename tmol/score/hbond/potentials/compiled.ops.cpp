#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/tuple.hh>

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
  static std::vector<Tensor> forward(
      AutogradContext* ctx,

      Tensor coords,
      Tensor block_pair_dispatch_indices,
      Tensor rot_to_res_map,
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
        coords.options(), "hbond_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              HBondPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
                  TCAST(coords),
                  TCAST(block_pair_dispatch_indices),
                  TCAST(rot_to_res_map),
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
           block_pair_dispatch_indices,
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

    return {score, block_neighbors};
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
      auto block_pair_dispatch_indices = saved[i++];
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
          coords.options(), "hbond_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = HBondPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    TCAST(coords),
                    TCAST(block_pair_dispatch_indices),
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
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),

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
class HBondPoseScoresOp2
    : public torch::autograd::Function<HBondPoseScoresOp2<DispatchMethod>> {
 public:
  static std::vector<Tensor> forward(
      AutogradContext* ctx,
      Tensor rot_coords,
      Tensor rot_coord_offset,
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
        rot_coords.options(), "hbond_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              HBondPoseScoreDispatch2<DispatchMethod, Dev, Real, Int>::forward(
                  TCAST(rot_coords),
                  TCAST(rot_coord_offset),
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
                  rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          block_neighbors = std::get<2>(result).tensor;
        }));

    if (output_block_pair_energies) {
      // save inputs for deriv call in backwards
      auto max_n_rots_per_pose_tp =
          TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);
      ctx->save_for_backward(
          {rot_coords,
           rot_coord_offset,
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
           block_neighbors

          });
    } else {
      score = score.squeeze(-1).squeeze(-1);  // remove final 2 "dummy" dims
      ctx->save_for_backward({dscore_dcoords});
    }

    return {score, block_neighbors};
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

      auto rot_coords = saved[i++];
      auto rot_coord_offset = saved[i++];
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
          rot_coords.options(), "hbond_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = HBondPoseScoreDispatch2<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    TCAST(rot_coords),
                    TCAST(rot_coord_offset),
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

    return {dV_d_pose_coords, torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> hbond_pose_scores_op(
    Tensor coords,
    Tensor block_pair_dispatch_indices,
    Tensor rot_to_res_map,
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
      block_pair_dispatch_indices,
      rot_to_res_map,
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

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> hbond_pose_scores_op_2(
    Tensor rot_coords,
    Tensor rot_coord_offset,
    Tensor first_rot_for_block,
    Tensor first_rot_block_type,
    // Tensor block_pair_dispatch_indices,
    Tensor block_ind_for_rot,
    Tensor pose_ind_for_rot,

    // Tensor rot_coord_offset,
    Tensor block_type_ind_for_rot,

    Tensor n_rots_for_pose,
    Tensor rot_offset_for_pose,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    int64_t max_n_rots_per_pose,
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
  return HBondPoseScoresOp2<DispatchMethod>::apply(
      rot_coords,
      rot_coord_offset,
      first_rot_for_block,
      first_rot_block_type,
      // block_pair_dispatch_indices,
      block_ind_for_rot,
      pose_ind_for_rot,

      // rot_coord_offset,
      block_type_ind_for_rot,

      n_rots_for_pose,
      rot_offset_for_pose,
      n_rots_for_block,
      rot_offset_for_block,
      max_n_rots_per_pose,
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
  m.def("hbond_pose_scores", &hbond_pose_scores_op<common::DeviceOperations>);
  m.def(
      "hbond_pose_scores_2", &hbond_pose_scores_op_2<common::DeviceOperations>);
}

}  // namespace potentials
}  // namespace hbond
}  // namespace score
}  // namespace tmol
