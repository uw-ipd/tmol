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

// MPS round-trip: TPack allocates CPU for MPS inputs; move output back.
static inline at::Tensor mps_to_dev(at::Tensor t, c10::Device dev) {
  return dev.is_mps() ? t.to(dev) : t;
}

template <template <tmol::Device> class DispatchMethod>
class ElecPoseScoreOp
    : public torch::autograd::Function<ElecPoseScoreOp<DispatchMethod>> {
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

    c10::Device orig_device = rot_coords.device();
    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "elec_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              ElecPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
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
                  TCAST(block_type_partial_charge),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_inter_repr_path_distance),

                  TCAST(block_type_intra_repr_path_distance),
                  TCAST(global_params),
                  output_block_pair_energies,
                  rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          block_neighbors = std::get<2>(result).tensor;
        }));

    score = mps_to_dev(score, orig_device);
    dscore_dcoords = mps_to_dev(dscore_dcoords, orig_device);
    block_neighbors = mps_to_dev(block_neighbors, orig_device);

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
      score = score.squeeze(-1).squeeze(-1);
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }

    return {score, block_neighbors};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_pose_coords;

    if (saved.size() == 2) {
      // single-score mode
      auto saved_grads = ctx->get_saved_variables();
      auto saved_grad = saved_grads[0];
      auto pose_ind_for_atom = saved_grads[1];

      auto atom_ingrads = grad_outputs[0].index_select(1, pose_ind_for_atom);

      while (atom_ingrads.dim() < saved_grad.dim()) {
        atom_ingrads = atom_ingrads.unsqueeze(-1);
      }
      dV_d_pose_coords = saved_grad * atom_ingrads;

    } else {
      // block-pair mode
      int i = 0;

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
      auto block_type_partial_charge = saved[i++];
      auto block_type_n_interblock_bonds = saved[i++];
      auto block_type_atoms_forming_chemical_bonds = saved[i++];
      auto block_type_inter_repr_path_distance = saved[i++];

      auto block_type_intra_repr_path_distance = saved[i++];
      auto global_params = saved[i++];
      auto block_neighbors = saved[i++];

      c10::Device orig_device = rot_coords.device();
      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "elec_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = ElecPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
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

      dV_d_pose_coords = mps_to_dev(dV_d_pose_coords, orig_device);
    }

    return {
        dV_d_pose_coords, torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(),  torch::Tensor(),

        torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(),  torch::Tensor(), torch::Tensor(),
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
class ElecRotamerScoreOp
    : public torch::autograd::Function<ElecRotamerScoreOp<DispatchMethod>> {
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
      Tensor block_type_partial_charge,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,
      Tensor block_type_inter_repr_path_distance,

      Tensor block_type_intra_repr_path_distance,
      Tensor global_params,
      bool output_block_pair_energies) {
    assert(output_block_pair_energies);
    at::Tensor score;
    at::Tensor dscore_dcoords;
    at::Tensor dispatch_inds;

    c10::Device orig_device = rot_coords.device();
    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "elec_rotamer_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              ElecRotamerScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
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
                  TCAST(block_type_partial_charge),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_inter_repr_path_distance),

                  TCAST(block_type_intra_repr_path_distance),
                  TCAST(global_params),
                  output_block_pair_energies,
                  rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          dispatch_inds = std::get<2>(result).tensor;
        }));

    score = mps_to_dev(score, orig_device);
    dscore_dcoords = mps_to_dev(dscore_dcoords, orig_device);
    dispatch_inds = mps_to_dev(dispatch_inds, orig_device);

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

           pose_stack_min_bond_separation,
           pose_stack_inter_block_bondsep,

           block_type_n_atoms,
           block_type_partial_charge,
           block_type_n_interblock_bonds,
           block_type_atoms_forming_chemical_bonds,
           block_type_inter_repr_path_distance,

           block_type_intra_repr_path_distance,
           global_params,
           dispatch_inds});
    } else {
      ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    }

    return {score, dispatch_inds};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_pose_coords;

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
      auto block_type_partial_charge = saved[i++];
      auto block_type_n_interblock_bonds = saved[i++];
      auto block_type_atoms_forming_chemical_bonds = saved[i++];
      auto block_type_inter_repr_path_distance = saved[i++];

      auto block_type_intra_repr_path_distance = saved[i++];
      auto global_params = saved[i++];
      auto dispatch_inds = saved[i++];

      c10::Device orig_device = rot_coords.device();
      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "elec_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = ElecRotamerScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
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
                    TCAST(block_type_partial_charge),
                    TCAST(block_type_n_interblock_bonds),
                    TCAST(block_type_atoms_forming_chemical_bonds),
                    TCAST(block_type_inter_repr_path_distance),

                    TCAST(block_type_intra_repr_path_distance),
                    TCAST(global_params),
                    TCAST(dispatch_inds),
                    TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));

      dV_d_pose_coords = mps_to_dev(dV_d_pose_coords, orig_device);
    }

    return {
        dV_d_pose_coords, torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(),  torch::Tensor(),

        torch::Tensor(),  torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(),

        torch::Tensor(),  torch::Tensor(), torch::Tensor(),
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> elec_pose_scores_op(
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
    Tensor block_type_partial_charge,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_inter_repr_path_distance,

    Tensor block_type_intra_repr_path_distance,
    Tensor global_params,
    bool output_block_pair_energies) {
  return ElecPoseScoreOp<DispatchMethod>::apply(
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
      block_type_partial_charge,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_inter_repr_path_distance,

      block_type_intra_repr_path_distance,
      global_params,
      output_block_pair_energies);
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> elec_rotamer_scores_op(
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
    Tensor block_type_partial_charge,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_inter_repr_path_distance,

    Tensor block_type_intra_repr_path_distance,
    Tensor global_params,
    bool output_block_pair_energies) {
  return ElecRotamerScoreOp<DispatchMethod>::apply(
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
      block_type_partial_charge,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_inter_repr_path_distance,

      block_type_intra_repr_path_distance,
      global_params,
      output_block_pair_energies);
}

// See https://stackoverflow.com/a/3221914
TORCH_LIBRARY(tmol_elec, m) {
  m.def("elec_pose_scores", &elec_pose_scores_op<common::DeviceOperations>);
  m.def(
      "elec_rotamer_scores", &elec_rotamer_scores_op<common::DeviceOperations>);
}

}  // namespace potentials
}  // namespace elec
}  // namespace score
}  // namespace tmol
