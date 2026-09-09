#include <torch/torch.h>
#include <torch/script.h>

#include <limits>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/whole_pose_scoring.hh>

#include "ljlk_pose_score.hh"
#include "ljlk_elec_pose_score.hh"
// #include "rotamer_pair_energy_lj.hh"
// #include "rotamer_pair_energy_lk.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

// Cache the mgpu::standard_context_t objects
// so as to avoid re-initializing them at each
// kernel launch
ContextManager mgr;

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
      Tensor block_type_is_ligand_fragment,

      Tensor type_params,
      Tensor global_params,
      double max_dis,  // host scalar; needed by detect-neighbors call
      bool output_block_pair_energies,
      Tensor shared_compact_block_neighbors) {
    at::Tensor score, dscore_dcoords, block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "ljlk_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              LJLKPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
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

                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_heavy_atoms_in_tile),

                  TCAST(block_type_heavy_atoms_in_tile),
                  TCAST(block_type_atom_types),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_path_distance),
                  TCAST(block_type_is_ligand_fragment),

                  TCAST(type_params),
                  TCAST(global_params),
                  (Real)max_dis,
                  TCAST(shared_compact_block_neighbors),
                  output_block_pair_energies,
                  rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          block_neighbors = shared_compact_block_neighbors.numel() != 0
                                ? shared_compact_block_neighbors
                                : std::get<2>(result).tensor;
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
           block_type_is_ligand_fragment,

           type_params,
           global_params,
           block_neighbors});
    } else {
      score = score.squeeze(-1).squeeze(-1);  // remove final 2 "dummy" dims
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
      auto saved_grad = saved[0];
      dV_d_pose_coords =
          common::accumulate_whole_pose_gradients(saved_grad, grad_outputs[0]);
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
      auto block_type_is_ligand_fragment = saved[i++];

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

                    TCAST(pose_stack_min_bond_separation),
                    TCAST(pose_stack_inter_block_bondsep),
                    TCAST(block_type_n_atoms),
                    TCAST(block_type_n_heavy_atoms_in_tile),

                    TCAST(block_type_heavy_atoms_in_tile),
                    TCAST(block_type_atom_types),
                    TCAST(block_type_n_interblock_bonds),
                    TCAST(block_type_atoms_forming_chemical_bonds),
                    TCAST(block_type_path_distance),
                    TCAST(block_type_is_ligand_fragment),

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

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod, bool weighted>
class LJLKAndElecPoseScoreOp
    : public torch::autograd::Function<
          LJLKAndElecPoseScoreOp<DispatchMethod, weighted>> {
 public:
  static std::vector<Tensor> forward(
      AutogradContext* ctx,
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
      Tensor block_type_atom_types,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,
      Tensor block_type_ljlk_path_distance,
      Tensor block_type_is_ligand_fragment,
      Tensor ljlk_type_params,
      Tensor ljlk_global_params,
      Tensor block_type_partial_charge,
      Tensor block_type_elec_inter_repr_path_distance,
      Tensor block_type_elec_intra_repr_path_distance,
      Tensor elec_global_params,
      Tensor shared_compact_block_neighbors,
      Tensor score_weights) {
    TORCH_CHECK(
        shared_compact_block_neighbors.numel() != 0,
        "fused LJ/LK + electrostatics requires compact block neighbors");
    if constexpr (weighted) {
      TORCH_CHECK(
          score_weights.dim() == 1 && score_weights.size(0) == 4,
          "weighted fused LJ/LK + electrostatics requires four score weights");
      TORCH_CHECK(
          score_weights.device() == rot_coords.device()
              && score_weights.scalar_type() == rot_coords.scalar_type(),
          "fused score weights must match coordinate dtype and device");
    }
    Tensor score;
    Tensor dscore_dcoords;
    using Int = int32_t;
#define TMOL_FUSED_SCORE_ARGS                                                \
  mgr, TCAST(rot_coords), TCAST(rot_coord_offset), TCAST(pose_ind_for_atom), \
      TCAST(first_rot_for_block), TCAST(first_rot_block_type),               \
      TCAST(block_ind_for_rot), TCAST(pose_ind_for_rot),                     \
      TCAST(block_type_ind_for_rot), TCAST(n_rots_for_pose),                 \
      TCAST(rot_offset_for_pose), TCAST(n_rots_for_block),                   \
      TCAST(rot_offset_for_block), max_n_rots_per_pose,                      \
      TCAST(pose_stack_min_bond_separation),                                 \
      TCAST(pose_stack_inter_block_bondsep), TCAST(block_type_n_atoms),      \
      TCAST(block_type_atom_types), TCAST(block_type_n_interblock_bonds),    \
      TCAST(block_type_atoms_forming_chemical_bonds),                        \
      TCAST(block_type_ljlk_path_distance),                                  \
      TCAST(block_type_is_ligand_fragment), TCAST(ljlk_type_params),         \
      TCAST(ljlk_global_params), TCAST(block_type_partial_charge),           \
      TCAST(block_type_elec_inter_repr_path_distance),                       \
      TCAST(block_type_elec_intra_repr_path_distance),                       \
      TCAST(elec_global_params), TCAST(shared_compact_block_neighbors)
    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "ljlk_elec_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
          auto result = [&]() {
            if constexpr (weighted) {
              return LJLKAndElecPoseScoreDispatch<
                  DispatchMethod,
                  Dev,
                  Real,
                  Int>::
                  forward_weighted(
                      TMOL_FUSED_SCORE_ARGS,
                      TCAST(score_weights),
                      rot_coords.requires_grad());
            } else {
              return LJLKAndElecPoseScoreDispatch<
                  DispatchMethod,
                  Dev,
                  Real,
                  Int>::
                  forward(TMOL_FUSED_SCORE_ARGS, rot_coords.requires_grad());
            }
          }();
          score = std::get<0>(result).tensor.squeeze(-1).squeeze(-1);
          dscore_dcoords = std::get<1>(result).tensor;
        }));
#undef TMOL_FUSED_SCORE_ARGS
    ctx->save_for_backward({dscore_dcoords, pose_ind_for_atom});
    return {score};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto const saved = ctx->get_saved_variables();
    tensor_list gradients(29);
    gradients[0] =
        common::accumulate_whole_pose_gradients(saved[0], grad_outputs[0]);
    return gradients;
  }
};

template <template <tmol::Device> class DispatchMethod>
class WeightedFusedScoreSumOp : public torch::autograd::Function<
                                    WeightedFusedScoreSumOp<DispatchMethod>> {
 public:
  static std::vector<Tensor> forward(
      AutogradContext* ctx,
      Tensor score_lanes,
      Tensor score_weights,
      int64_t fused_weight_begin,
      int64_t fused_weight_width) {
    TORCH_CHECK(
        score_lanes.dim() == 2 && score_weights.dim() == 2
            && score_weights.size(1) == 1,
        "weighted score reduction expects [lane, pose] scores and [lane, 1] "
        "weights");
    TORCH_CHECK(
        score_lanes.device() == score_weights.device()
            && score_lanes.scalar_type() == score_weights.scalar_type(),
        "score lanes and weights must have the same dtype and device");
    TORCH_CHECK(
        fused_weight_begin >= 0 && fused_weight_width > 0
            && fused_weight_begin < score_lanes.size(0)
            && score_lanes.size(0)
                   == score_weights.size(0) - fused_weight_width + 1,
        "invalid fused score-lane mapping");

    Tensor output;
    using Int = int32_t;
    TMOL_DISPATCH_FLOATING_DEVICE(
        score_lanes.options(), "weighted_fused_score_sum_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
          output =
              LJLKAndElecPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  reduce_weighted_scores(
                      mgr,
                      TCAST(score_lanes),
                      TCAST(score_weights),
                      fused_weight_begin,
                      fused_weight_width)
                      .tensor;
        }));
    ctx->save_for_backward({score_weights});
    ctx->saved_data["n_score_lanes"] = score_lanes.size(0);
    ctx->saved_data["fused_weight_begin"] = fused_weight_begin;
    ctx->saved_data["fused_weight_width"] = fused_weight_width;
    return {output};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    Tensor const score_weights = ctx->get_saved_variables()[0];
    // TView preserves the incoming stride, including the zero stride produced
    // by score.sum()'s expanded gradient. Avoid materializing that expansion:
    // the native reduction reads it correctly and saves one CPU copy or CUDA
    // kernel launch from every default score-plus-gradient call.
    Tensor const output_gradient = grad_outputs[0];
    int64_t const n_score_lanes = ctx->saved_data["n_score_lanes"].toInt();
    int64_t const fused_weight_begin =
        ctx->saved_data["fused_weight_begin"].toInt();
    int64_t const fused_weight_width =
        ctx->saved_data["fused_weight_width"].toInt();
    Tensor score_lane_gradients;
    using Int = int32_t;
    TMOL_DISPATCH_FLOATING_DEVICE(
        output_gradient.options(), "weighted_fused_score_sum_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;
          score_lane_gradients =
              LJLKAndElecPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  reduce_weighted_score_gradients(
                      mgr,
                      TCAST(output_gradient),
                      TCAST(score_weights),
                      n_score_lanes,
                      fused_weight_begin,
                      fused_weight_width)
                      .tensor;
        }));
    tensor_list gradients(4);
    gradients[0] = score_lane_gradients;
    return gradients;
  }
};

template <template <tmol::Device> class DispatchMethod>
class LJLKRotamerScoreOp
    : public torch::autograd::Function<LJLKRotamerScoreOp<DispatchMethod>> {
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
      Tensor block_type_is_ligand_fragment,

      Tensor type_params,
      Tensor global_params,
      double max_dis,  // host scalar; needed by detect-neighbors call
      bool output_block_pair_energies) {
    at::Tensor score, dscore_dcoords, dispatch_indices;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "ljlk_rotamer_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              LJLKRotamerScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
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

                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_heavy_atoms_in_tile),

                  TCAST(block_type_heavy_atoms_in_tile),
                  TCAST(block_type_atom_types),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),
                  TCAST(block_type_path_distance),
                  TCAST(block_type_is_ligand_fragment),

                  TCAST(type_params),
                  TCAST(global_params),
                  (Real)max_dis,
                  output_block_pair_energies,
                  rot_coords.requires_grad());

          score = std::get<0>(result).tensor;
          dscore_dcoords = std::get<1>(result).tensor;
          dispatch_indices = std::get<2>(result).tensor;
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
           block_type_is_ligand_fragment,

           type_params,
           global_params,
           dispatch_indices});
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
      auto atom_ingrads =
          grad_outputs[0].index_select(1, pose_ind_for_atom).unsqueeze(-1);

      dV_d_pose_coords = saved_grad * atom_ingrads;
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
      auto block_type_is_ligand_fragment = saved[i++];

      auto type_params = saved[i++];
      auto global_params = saved[i++];
      auto dispatch_indices = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "ljlk_rotamer_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = LJLKRotamerScoreDispatch<
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

                    TCAST(pose_stack_min_bond_separation),
                    TCAST(pose_stack_inter_block_bondsep),
                    TCAST(block_type_n_atoms),
                    TCAST(block_type_n_heavy_atoms_in_tile),

                    TCAST(block_type_heavy_atoms_in_tile),
                    TCAST(block_type_atom_types),
                    TCAST(block_type_n_interblock_bonds),
                    TCAST(block_type_atoms_forming_chemical_bonds),
                    TCAST(block_type_path_distance),
                    TCAST(block_type_is_ligand_fragment),

                    TCAST(type_params),
                    TCAST(global_params),
                    TCAST(dispatch_indices),
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

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor()};
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
    Tensor block_type_is_ligand_fragment,

    Tensor ljlk_type_params,
    Tensor global_params,
    double max_dis,
    bool output_block_pair_energies,
    Tensor shared_compact_block_neighbors) {
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
      block_type_is_ligand_fragment,

      ljlk_type_params,
      global_params,
      max_dis,
      output_block_pair_energies,
      shared_compact_block_neighbors);
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> ljlk_elec_pose_scores_op(
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
    Tensor block_type_atom_types,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_ljlk_path_distance,
    Tensor block_type_is_ligand_fragment,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor block_type_partial_charge,
    Tensor block_type_elec_inter_repr_path_distance,
    Tensor block_type_elec_intra_repr_path_distance,
    Tensor elec_global_params,
    Tensor shared_compact_block_neighbors) {
  return LJLKAndElecPoseScoreOp<DispatchMethod, false>::apply(
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
      block_type_atom_types,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_ljlk_path_distance,
      block_type_is_ligand_fragment,
      ljlk_type_params,
      ljlk_global_params,
      block_type_partial_charge,
      block_type_elec_inter_repr_path_distance,
      block_type_elec_intra_repr_path_distance,
      elec_global_params,
      shared_compact_block_neighbors,
      torch::empty({0}, rot_coords.options()));
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> ljlk_elec_weighted_pose_scores_op(
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
    Tensor block_type_atom_types,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_ljlk_path_distance,
    Tensor block_type_is_ligand_fragment,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor block_type_partial_charge,
    Tensor block_type_elec_inter_repr_path_distance,
    Tensor block_type_elec_intra_repr_path_distance,
    Tensor elec_global_params,
    Tensor shared_compact_block_neighbors,
    Tensor score_weights) {
  return LJLKAndElecPoseScoreOp<DispatchMethod, true>::apply(
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
      block_type_atom_types,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_ljlk_path_distance,
      block_type_is_ligand_fragment,
      ljlk_type_params,
      ljlk_global_params,
      block_type_partial_charge,
      block_type_elec_inter_repr_path_distance,
      block_type_elec_intra_repr_path_distance,
      elec_global_params,
      shared_compact_block_neighbors,
      score_weights);
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> weighted_fused_score_sum_op(
    Tensor score_lanes,
    Tensor score_weights,
    int64_t fused_weight_begin,
    int64_t fused_weight_width) {
  return WeightedFusedScoreSumOp<DispatchMethod>::apply(
      score_lanes, score_weights, fused_weight_begin, fused_weight_width);
}

std::vector<Tensor> build_compact_block_neighbors_op(
    Tensor rot_coords,
    Tensor rot_coord_offset,
    Tensor first_rot_block_type,
    Tensor block_ind_for_rot,
    Tensor pose_ind_for_rot,
    Tensor block_type_ind_for_rot,
    Tensor block_type_n_atoms,
    double reach) {
  TORCH_CHECK(
      first_rot_block_type.dim() == 2,
      "first_rot_block_type must have shape [n_poses, max_n_blocks]");
  int64_t const n_poses = first_rot_block_type.size(0);
  int64_t const max_n_blocks = first_rot_block_type.size(1);
  int64_t constexpr max_int = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(
      n_poses <= max_int,
      "compact block-neighbor indexing supports at most ",
      max_int,
      " poses; got ",
      n_poses);
  TORCH_CHECK(
      max_n_blocks <= 65535,
      "compact block-neighbor indexing supports at most 65,535 blocks per "
      "pose; got ",
      max_n_blocks);
  int64_t const pairs_per_pose = max_n_blocks * (max_n_blocks + 1) / 2;
  TORCH_CHECK(
      pairs_per_pose == 0 || n_poses <= (max_int - 1) / pairs_per_pose,
      "compact block-neighbor indexing exceeds int32 capacity for ",
      n_poses,
      " poses and ",
      max_n_blocks,
      " blocks per pose");
  Tensor neighbor_indices;
  using Int = int32_t;
  TMOL_DISPATCH_FLOATING_DEVICE(
      rot_coords.options(), "build_compact_block_neighbors", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        neighbor_indices =
            LJLKPoseScoreDispatch<DeviceOperations, Dev, Real, Int>::
                build_compact_block_neighbors(
                    mgr,
                    TCAST(rot_coords),
                    TCAST(rot_coord_offset),
                    TCAST(first_rot_block_type),
                    TCAST(block_ind_for_rot),
                    TCAST(pose_ind_for_rot),
                    TCAST(block_type_ind_for_rot),
                    TCAST(block_type_n_atoms),
                    (Real)reach)
                    .tensor;
      }));
  return {neighbor_indices};
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> ljlk_rotamer_scores_op(
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
    Tensor block_type_is_ligand_fragment,

    Tensor ljlk_type_params,
    Tensor global_params,
    double max_dis,
    bool output_block_pair_energies) {
  return LJLKRotamerScoreOp<DispatchMethod>::apply(
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
      block_type_is_ligand_fragment,

      ljlk_type_params,
      global_params,
      max_dis,
      output_block_pair_energies);
}

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> ljlk_elec_weighted_rotamer_scores_op(
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
    Tensor block_type_atom_types,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_ljlk_path_distance,
    Tensor block_type_is_ligand_fragment,
    Tensor ljlk_type_params,
    Tensor ljlk_global_params,
    Tensor block_type_partial_charge,
    Tensor block_type_elec_inter_repr_path_distance,
    Tensor block_type_elec_intra_repr_path_distance,
    Tensor elec_global_params,
    double max_dis,
    Tensor score_weights,
    Tensor output_gradients,
    Tensor shared_dispatch_indices) {
  TORCH_CHECK(
      !torch::GradMode::is_enabled()
          || (!rot_coords.requires_grad() && !score_weights.requires_grad()),
      "weighted fused rotamer scoring requires the autograd module wrapper");
  TORCH_CHECK(
      score_weights.dim() == 1 && score_weights.size(0) == 4,
      "weighted fused rotamer scoring requires four score weights");
  TORCH_CHECK(
      score_weights.scalar_type() == rot_coords.scalar_type()
          && score_weights.device() == rot_coords.device(),
      "fused rotamer weights must match coordinate dtype and device");
  TORCH_CHECK(
      output_gradients.numel() == 0
          || (output_gradients.dim() == 1
              && output_gradients.scalar_type() == rot_coords.scalar_type()
              && output_gradients.device() == rot_coords.device()),
      "fused rotamer output gradients must be an empty or matching vector");
  TORCH_CHECK(
      shared_dispatch_indices.dim() == 2
          && (shared_dispatch_indices.size(0) == 3
              || (shared_dispatch_indices.size(0) == 0
                  && shared_dispatch_indices.size(1) == 0)),
      "shared rotamer dispatch indices must have shape [3, nnz] or [0, 0]");
  TORCH_CHECK(
      shared_dispatch_indices.scalar_type() == torch::kInt32,
      "shared rotamer dispatch indices must have dtype int32");
  TORCH_CHECK(
      shared_dispatch_indices.device() == rot_coords.device(),
      "shared rotamer dispatch indices must be on the coordinate device");
  TORCH_CHECK(
      (shared_dispatch_indices.size(0) == 0 && output_gradients.numel() == 0)
          || (shared_dispatch_indices.size(0) == 3
              && output_gradients.numel() == shared_dispatch_indices.size(1)),
      "fused rotamer output gradients require the matching forward dispatch");

  Tensor score, dscore_dcoords, dispatch_indices;
  using Int = int32_t;
  TMOL_DISPATCH_FLOATING_DEVICE(
      rot_coords.options(), "ljlk_elec_weighted_rotamer_scores", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;
        auto result =
            LJLKAndElecPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                forward_weighted_rotamers(
                    mgr,
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
                    TCAST(block_type_atom_types),
                    TCAST(block_type_n_interblock_bonds),
                    TCAST(block_type_atoms_forming_chemical_bonds),
                    TCAST(block_type_ljlk_path_distance),
                    TCAST(block_type_is_ligand_fragment),
                    TCAST(ljlk_type_params),
                    TCAST(ljlk_global_params),
                    TCAST(block_type_partial_charge),
                    TCAST(block_type_elec_inter_repr_path_distance),
                    TCAST(block_type_elec_intra_repr_path_distance),
                    TCAST(elec_global_params),
                    (Real)max_dis,
                    TCAST(score_weights),
                    TCAST(output_gradients),
                    TCAST(shared_dispatch_indices));
        score = std::get<0>(result).tensor.squeeze(1).squeeze(1);
        dscore_dcoords = std::get<1>(result).tensor.squeeze(0);
        dispatch_indices = std::get<2>(result).tensor;
      }));
  TORCH_CHECK(
      output_gradients.numel() == 0
          || output_gradients.numel() == score.numel(),
      "fused rotamer output gradients must match the score table");
  return {score, dispatch_indices, dscore_dcoords};
}

// See https://stackoverflow.com/a/3221914
TORCH_LIBRARY(tmol_ljlk, m) {
  m.def("ljlk_pose_scores", &ljlk_pose_scores_op<DeviceOperations>);
  m.def("ljlk_elec_pose_scores", &ljlk_elec_pose_scores_op<DeviceOperations>);
  m.def(
      "ljlk_elec_weighted_pose_scores",
      &ljlk_elec_weighted_pose_scores_op<DeviceOperations>);
  m.def(
      "weighted_fused_score_sum",
      &weighted_fused_score_sum_op<DeviceOperations>);
  m.def("ljlk_rotamer_scores", &ljlk_rotamer_scores_op<DeviceOperations>);
  m.def(
      "ljlk_elec_weighted_rotamer_scores",
      &ljlk_elec_weighted_rotamer_scores_op<DeviceOperations>);
  m.def("build_compact_block_neighbors", &build_compact_block_neighbors_op);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
