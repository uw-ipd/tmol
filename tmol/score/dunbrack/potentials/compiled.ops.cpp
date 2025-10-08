#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "dunbrack_pose_score.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using namespace tmol::score::common;

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

template <template <tmol::Device> class DispatchMethod>
class DunbrackPoseScoreOp
    : public torch::autograd::Function<DunbrackPoseScoreOp<DispatchMethod>> {
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
      Tensor block_type_atom_downstream_of_conn,

      Tensor rotameric_neglnprob_tables,
      Tensor rotprob_table_sizes,
      Tensor rotprob_table_strides,
      Tensor rotameric_mean_tables,
      Tensor rotameric_sdev_tables,
      Tensor rotmean_table_sizes,
      Tensor rotmean_table_strides,

      Tensor rotameric_bb_start,
      Tensor rotameric_bb_step,
      Tensor rotameric_bb_periodicity,

      Tensor rotameric_rotind2tableind,
      Tensor semirotameric_rotind2tableind,

      Tensor semirotameric_tables,
      Tensor semirot_table_sizes,
      Tensor semirot_table_strides,
      Tensor semirot_start,
      Tensor semirot_step,
      Tensor semirot_periodicity,

      Tensor res_n_dihedrals,
      Tensor res_dih_uaids,
      Tensor res_rotamer_table_set,
      Tensor res_rotameric_index,
      Tensor res_semirotameric_index,
      Tensor res_n_chi,
      Tensor res_n_rotameric_chi,
      Tensor res_probability_table_offset,
      Tensor res_mean_table_offset,
      Tensor res_rotamer_index_to_table_index,
      Tensor block_semirotameric_tableset_offset,
      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dscore_dcoords;
    at::Tensor block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "dunbrack_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              DunbrackPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
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
                      TCAST(block_type_atom_downstream_of_conn),

                      TCAST(rotameric_neglnprob_tables),
                      TCAST(rotprob_table_sizes),
                      TCAST(rotprob_table_strides),
                      TCAST(rotameric_mean_tables),
                      TCAST(rotameric_sdev_tables),
                      TCAST(rotmean_table_sizes),
                      TCAST(rotmean_table_strides),

                      TCAST(rotameric_bb_start),
                      TCAST(rotameric_bb_step),
                      TCAST(rotameric_bb_periodicity),

                      TCAST(rotameric_rotind2tableind),
                      TCAST(semirotameric_rotind2tableind),

                      TCAST(semirotameric_tables),
                      TCAST(semirot_table_sizes),
                      TCAST(semirot_table_strides),
                      TCAST(semirot_start),
                      TCAST(semirot_step),
                      TCAST(semirot_periodicity),

                      TCAST(res_n_dihedrals),
                      TCAST(res_dih_uaids),
                      TCAST(res_rotamer_table_set),
                      TCAST(res_rotameric_index),
                      TCAST(res_semirotameric_index),
                      TCAST(res_n_chi),
                      TCAST(res_n_rotameric_chi),
                      TCAST(res_probability_table_offset),
                      TCAST(res_mean_table_offset),
                      TCAST(res_rotamer_index_to_table_index),
                      TCAST(block_semirotameric_tableset_offset),
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

           pose_stack_inter_block_connections,
           block_type_atom_downstream_of_conn,

           rotameric_neglnprob_tables,
           rotprob_table_sizes,
           rotprob_table_strides,
           rotameric_mean_tables,
           rotameric_sdev_tables,
           rotmean_table_sizes,
           rotmean_table_strides,

           rotameric_bb_start,
           rotameric_bb_step,
           rotameric_bb_periodicity,

           rotameric_rotind2tableind,
           semirotameric_rotind2tableind,

           semirotameric_tables,
           semirot_table_sizes,
           semirot_table_strides,
           semirot_start,
           semirot_step,
           semirot_periodicity,

           res_n_dihedrals,
           res_dih_uaids,
           res_rotamer_table_set,
           res_rotameric_index,
           res_semirotameric_index,
           res_n_chi,
           res_n_rotameric_chi,
           res_probability_table_offset,
           res_mean_table_offset,
           res_rotamer_index_to_table_index,
           block_semirotameric_tableset_offset});
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
      // whole-pose-score mode
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
      auto block_type_atom_downstream_of_conn = saved[i++];

      auto rotameric_neglnprob_tables = saved[i++];
      auto rotprob_table_sizes = saved[i++];
      auto rotprob_table_strides = saved[i++];
      auto rotameric_mean_tables = saved[i++];
      auto rotameric_sdev_tables = saved[i++];
      auto rotmean_table_sizes = saved[i++];
      auto rotmean_table_strides = saved[i++];

      auto rotameric_bb_start = saved[i++];
      auto rotameric_bb_step = saved[i++];
      auto rotameric_bb_periodicity = saved[i++];

      auto rotameric_rotind2tableind = saved[i++];
      auto semirotameric_rotind2tableind = saved[i++];

      auto semirotameric_tables = saved[i++];
      auto semirot_table_sizes = saved[i++];
      auto semirot_table_strides = saved[i++];
      auto semirot_start = saved[i++];
      auto semirot_step = saved[i++];
      auto semirot_periodicity = saved[i++];

      auto res_n_dihedrals = saved[i++];
      auto res_dih_uaids = saved[i++];
      auto res_rotamer_table_set = saved[i++];
      auto res_rotameric_index = saved[i++];
      auto res_semirotameric_index = saved[i++];
      auto res_n_chi = saved[i++];
      auto res_n_rotameric_chi = saved[i++];
      auto res_probability_table_offset = saved[i++];
      auto res_mean_table_offset = saved[i++];
      auto res_rotamer_index_to_table_index = saved[i++];
      auto block_semirotameric_tableset_offset = saved[i++];

      using Int = int32_t;

      auto dTdV = grad_outputs[0];

      TMOL_DISPATCH_FLOATING_DEVICE(
          rot_coords.options(), "dunbrack_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = DunbrackPoseScoreDispatch<
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

                    TCAST(pose_stack_inter_block_connections),
                    TCAST(block_type_atom_downstream_of_conn),

                    TCAST(rotameric_neglnprob_tables),
                    TCAST(rotprob_table_sizes),
                    TCAST(rotprob_table_strides),
                    TCAST(rotameric_mean_tables),
                    TCAST(rotameric_sdev_tables),
                    TCAST(rotmean_table_sizes),
                    TCAST(rotmean_table_strides),

                    TCAST(rotameric_bb_start),
                    TCAST(rotameric_bb_step),
                    TCAST(rotameric_bb_periodicity),

                    TCAST(rotameric_rotind2tableind),
                    TCAST(semirotameric_rotind2tableind),

                    TCAST(semirotameric_tables),
                    TCAST(semirot_table_sizes),
                    TCAST(semirot_table_strides),
                    TCAST(semirot_start),
                    TCAST(semirot_step),
                    TCAST(semirot_periodicity),

                    TCAST(res_n_dihedrals),
                    TCAST(res_dih_uaids),
                    TCAST(res_rotamer_table_set),
                    TCAST(res_rotameric_index),
                    TCAST(res_semirotameric_index),
                    TCAST(res_n_chi),
                    TCAST(res_n_rotameric_chi),
                    TCAST(res_probability_table_offset),
                    TCAST(res_mean_table_offset),
                    TCAST(res_rotamer_index_to_table_index),
                    TCAST(block_semirotameric_tableset_offset),
                    TCAST(dTdV));

            dV_d_pose_coords = result.tensor;
          }));
    }

    return {dV_d_pose_coords,

            // 12 common params in addition to rot_coords
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(),


            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod>
std::vector<Tensor> dunbrack_pose_scores_op(
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
    Tensor block_type_atom_downstream_of_conn,

    Tensor rotameric_neglnprob_tables,
    Tensor rotprob_table_sizes,
    Tensor rotprob_table_strides,
    Tensor rotameric_mean_tables,
    Tensor rotameric_sdev_tables,
    Tensor rotmean_table_sizes,
    Tensor rotmean_table_strides,

    Tensor rotameric_bb_start,
    Tensor rotameric_bb_step,
    Tensor rotameric_bb_periodicity,

    Tensor rotameric_rotind2tableind,
    Tensor semirotameric_rotind2tableind,

    Tensor semirotameric_tables,
    Tensor semirot_table_sizes,
    Tensor semirot_table_strides,
    Tensor semirot_start,
    Tensor semirot_step,
    Tensor semirot_periodicity,

    Tensor res_n_dihedrals,
    Tensor res_dih_uaids,
    Tensor res_rotamer_table_set,
    Tensor res_rotameric_index,
    Tensor res_semirotameric_index,
    Tensor res_n_chi,
    Tensor res_n_rotameric_chi,
    Tensor res_probability_table_offset,
    Tensor res_mean_table_offset,
    Tensor res_rotamer_index_to_table_index,
    Tensor block_semirotameric_tableset_offset,
    bool output_block_pair_energies) {
  return DunbrackPoseScoreOp<DispatchMethod>::apply(
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
      block_type_atom_downstream_of_conn,

      rotameric_neglnprob_tables,
      rotprob_table_sizes,
      rotprob_table_strides,
      rotameric_mean_tables,
      rotameric_sdev_tables,
      rotmean_table_sizes,
      rotmean_table_strides,

      rotameric_bb_start,
      rotameric_bb_step,
      rotameric_bb_periodicity,

      rotameric_rotind2tableind,
      semirotameric_rotind2tableind,

      semirotameric_tables,
      semirot_table_sizes,
      semirot_table_strides,
      semirot_start,
      semirot_step,
      semirot_periodicity,

      res_n_dihedrals,
      res_dih_uaids,
      res_rotamer_table_set,
      res_rotameric_index,
      res_semirotameric_index,
      res_n_chi,
      res_n_rotameric_chi,
      res_probability_table_offset,
      res_mean_table_offset,
      res_rotamer_index_to_table_index,
      block_semirotameric_tableset_offset,
      output_block_pair_energies);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("dunbrack_pose_scores", &dunbrack_pose_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
