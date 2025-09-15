#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "ljlk_pose_score.hh"
#include "rotamer_pair_energy_lj.hh"
// #include "rotamer_pair_energy_lk.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class LJLKPoseScoreOp
    : public torch::autograd::Function<LJLKPoseScoreOp<DispatchMethod>> {
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
      Tensor global_params,
      bool output_block_pair_energies) {
    at::Tensor score, dscore_dcoords, block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "ljlk_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              LJLKPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
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

           pose_stack_min_bond_separation,
           pose_stack_inter_block_bondsep,
           block_type_n_atoms,
           block_type_n_heavy_atoms_in_tile,

           block_type_heavy_atoms_in_tile,
           block_type_atom_types,
           block_type_n_interblock_bonds,
           block_type_atoms_forming_chemical_bonds,
           block_type_path_distance,

           type_params,
           global_params,
           block_neighbors});
    } else {
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }

    return {score, block_neighbors};
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

      printf("SAVED_GRAD\n");
      for (int ii = 0; ii < saved_grad.dim(); ii++)
        printf("%i\n", saved_grad.size(ii));

      printf("ATOM_INGRAD\n");
      for (int ii = 0; ii < atom_ingrads.dim(); ii++)
        printf("%i\n", atom_ingrads.size(ii));

      result.emplace_back(saved_grad * atom_ingrads);
      printf("post mult grad\n");
      //}

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

      auto pose_stack_min_bond_separation = saved[i++];
      auto pose_stack_inter_block_bondsep = saved[i++];
      auto block_type_n_atoms = saved[i++];
      auto block_type_n_heavy_atoms_in_tile = saved[i++];

      auto block_type_heavy_atoms_in_tile = saved[i++];
      auto block_type_atom_types = saved[i++];
      auto block_type_n_interblock_bonds = saved[i++];
      auto block_type_atoms_forming_chemical_bonds = saved[i++];
      auto block_type_path_distance = saved[i++];

      auto type_params = saved[i++];
      auto global_params = saved[i++];
      auto block_neighbors = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "ljlk_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = LJLKPoseScoreDispatch<
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
                    TCAST(block_neighbors),
                    TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));
    }

    return {dV_d_pose_coords, torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> ljlk_pose_scores_op(
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
    Tensor global_params,
    bool output_block_pair_energies) {
  return LJLKPoseScoreOp<DispatchMethod>::apply(
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
      global_params,
      output_block_pair_energies);
}

Tensor rotamer_pair_energies_op(
    Tensor context_coords,
    Tensor context_coord_offsets,
    Tensor context_block_type,
    Tensor alternate_coords,
    Tensor alternate_coord_offsets,
    Tensor alternate_ids,
    Tensor context_system_ids,
    Tensor system_min_bond_separation,
    Tensor system_inter_block_bondsep,
    Tensor system_neighbor_list,
    Tensor block_type_n_atoms,
    Tensor block_type_n_heavy_atoms,
    Tensor block_type_n_heavy_atoms_in_tile,
    Tensor block_type_heavy_atoms_in_tile,
    Tensor block_type_atom_types,
    Tensor block_type_heavy_atom_inds,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_path_distance,
    Tensor ljlk_type_params,
    Tensor global_params,
    Tensor lj_lk_weights) {
  using Int = int32_t;
  Tensor output_tensor;

  auto empty_score_event_tensor =
      TPack<int64_t, 1, tmol::Device::CPU>::zeros({1});
  auto empty_annealer_event_tensor =
      TPack<int64_t, 1, tmol::Device::CPU>::zeros({1});

  TMOL_DISPATCH_FLOATING_DEVICE(
      context_coords.options(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto output_tp =
            TPack<Real, 1, Dev>::zeros({alternate_coord_offsets.size(0)});
        auto output_tv = output_tp.view;

        LJLKRPEDispatch<common::ForallDispatch, Dev, Real, Int>::f(
            TCAST(context_coords),
            TCAST(context_coord_offsets),
            TCAST(context_block_type),
            TCAST(alternate_coords),
            TCAST(alternate_coord_offsets),
            TCAST(alternate_ids),
            TCAST(context_system_ids),
            TCAST(system_min_bond_separation),
            TCAST(system_inter_block_bondsep),
            TCAST(system_neighbor_list),
            TCAST(block_type_n_atoms),
            TCAST(block_type_n_heavy_atoms_in_tile),
            TCAST(block_type_heavy_atoms_in_tile),
            TCAST(block_type_atom_types),
            TCAST(block_type_n_interblock_bonds),
            TCAST(block_type_atoms_forming_chemical_bonds),
            TCAST(block_type_path_distance),
            TCAST(ljlk_type_params),
            TCAST(global_params),
            TCAST(lj_lk_weights),
            output_tv,
            empty_score_event_tensor.view,
            empty_annealer_event_tensor.view);

        output_tensor = output_tp.tensor;
      }));

  return output_tensor;
}

Tensor register_lj_lk_rotamer_pair_energy_eval(
    Tensor context_coords,
    Tensor context_coord_offsets,
    Tensor context_block_type,
    Tensor alternate_coords,
    Tensor alternate_coord_offsets,
    Tensor alternate_ids,
    Tensor context_system_ids,
    Tensor system_min_bond_separation,
    Tensor system_inter_block_bondsep,
    Tensor system_neighbor_list,
    Tensor block_type_n_atoms,
    Tensor block_type_n_heavy_atoms,
    Tensor block_type_n_heavy_atoms_in_tile,
    Tensor block_type_heavy_atoms_in_tile,
    Tensor block_type_atom_types,
    Tensor block_type_heavy_atom_inds,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_path_distance,
    Tensor ljlk_type_params,
    Tensor global_params,
    Tensor lj_lk_weights,
    Tensor output,
    Tensor score_event,
    Tensor annealer_event,
    Tensor annealer) {
  Tensor dummy_return_value;
  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      context_coords.options(), "score_op", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        LJLKRPERegistratorDispatch<common::ForallDispatch, Dev, Real, Int>::f(
            TCAST(context_coords),
            TCAST(context_coord_offsets),
            TCAST(context_block_type),
            TCAST(alternate_coords),
            TCAST(alternate_coord_offsets),
            TCAST(alternate_ids),
            TCAST(context_system_ids),
            TCAST(system_min_bond_separation),
            TCAST(system_inter_block_bondsep),
            TCAST(system_neighbor_list),
            TCAST(block_type_n_atoms),
            TCAST(block_type_n_heavy_atoms_in_tile),
            TCAST(block_type_heavy_atoms_in_tile),
            TCAST(block_type_atom_types),
            TCAST(block_type_n_interblock_bonds),
            TCAST(block_type_atoms_forming_chemical_bonds),
            TCAST(block_type_path_distance),
            TCAST(ljlk_type_params),
            TCAST(global_params),
            TCAST(lj_lk_weights),
            TCAST(output),
            TCAST(score_event),
            TCAST(annealer_event),
            TCAST(annealer));
      }));
  return dummy_return_value;
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("ljlk_pose_scores", &ljlk_pose_scores_op<DeviceOperations>);
  m.def("score_ljlk_inter_system_scores", &rotamer_pair_energies_op);
  m.def(
      "register_lj_lk_rotamer_pair_energy_eval",
      &register_lj_lk_rotamer_pair_energy_eval);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
