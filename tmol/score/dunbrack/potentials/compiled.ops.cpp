#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "dispatch.hh"
#include "dunbrack_pose_score.hh"

namespace tmol {
namespace score {
namespace dunbrack {
namespace potentials {

using namespace tmol::score::common;

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
class DunOp : public Function<DunOp<ScoreDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
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
      Tensor semirotameric_tables,
      Tensor semirot_table_sizes,
      Tensor semirot_table_strides,
      Tensor semirot_start,
      Tensor semirot_step,
      Tensor semirot_periodicity,
      Tensor rotameric_rotind2tableind,
      Tensor semirotameric_rotind2tableind,
      Tensor ndihe_for_res,
      Tensor dihedral_offset_for_res,
      Tensor dihedral_atom_inds,
      Tensor rottable_set_for_res,
      Tensor nchi_for_res,
      Tensor nrotameric_chi_for_res,
      Tensor rotres2resid,
      Tensor prob_table_offset_for_rotresidue,
      Tensor rotind2tableind_offset_for_res,
      Tensor rotmean_table_offset_for_residue,
      Tensor rotameric_chi_desc,
      Tensor semirotameric_chi_desc,
      Tensor dihedrals,
      Tensor ddihe_dxyz,
      Tensor rotameric_rottable_assignment,
      Tensor semirotameric_rottable_assignment) {
    at::Tensor score;
    at::Tensor dneglnprob_rot_dbb_xyz;
    at::Tensor drotchi_devpen_dtor_xyz;
    at::Tensor dneglnprob_nonrot_dtor_xyz;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "dun_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              DunbrackDispatch<DispatchMethod, Dev, Real, Int>::forward(
                  TCAST(coords),
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
                  TCAST(semirotameric_tables),
                  TCAST(semirot_table_sizes),
                  TCAST(semirot_table_strides),
                  TCAST(semirot_start),
                  TCAST(semirot_step),
                  TCAST(semirot_periodicity),
                  TCAST(rotameric_rotind2tableind),
                  TCAST(semirotameric_rotind2tableind),
                  TCAST(ndihe_for_res),
                  TCAST(dihedral_offset_for_res),
                  TCAST(dihedral_atom_inds),
                  TCAST(rottable_set_for_res),
                  TCAST(nchi_for_res),
                  TCAST(nrotameric_chi_for_res),
                  TCAST(rotres2resid),
                  TCAST(prob_table_offset_for_rotresidue),
                  TCAST(rotind2tableind_offset_for_res),
                  TCAST(rotmean_table_offset_for_residue),
                  TCAST(rotameric_chi_desc),
                  TCAST(semirotameric_chi_desc),
                  TCAST(dihedrals),
                  TCAST(ddihe_dxyz),
                  TCAST(rotameric_rottable_assignment),
                  TCAST(semirotameric_rottable_assignment));

          score = std::get<0>(result).tensor;

          // save for backwards
          dneglnprob_rot_dbb_xyz = std::get<1>(result).tensor;
          drotchi_devpen_dtor_xyz = std::get<2>(result).tensor;
          dneglnprob_nonrot_dtor_xyz = std::get<3>(result).tensor;
        }));

    ctx->save_for_backward(
        {coords,
         dneglnprob_rot_dbb_xyz,
         drotchi_devpen_dtor_xyz,
         dneglnprob_nonrot_dtor_xyz,
         dihedral_offset_for_res,
         dihedral_atom_inds,
         rotres2resid,
         rotameric_chi_desc,
         semirotameric_chi_desc});

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    int i = 0;

    auto coords = saved[i++];
    auto dneglnprob_rot_dbb_xyz = saved[i++];
    auto drotchi_devpen_dtor_xyz = saved[i++];
    auto dneglnprob_nonrot_dtor_xyz = saved[i++];
    auto dihedral_offset_for_res = saved[i++];
    auto dihedral_atom_inds = saved[i++];
    auto rotres2resid = saved[i++];
    auto rotameric_chi_desc = saved[i++];
    auto semirotameric_chi_desc = saved[i++];

    using Int = int32_t;

    at::Tensor dT_dI;
    auto dTdV = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "ScoreOpBackward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::backward(
              TCAST(dTdV),
              TCAST(coords),
              TCAST(dneglnprob_rot_dbb_xyz),
              TCAST(drotchi_devpen_dtor_xyz),
              TCAST(dneglnprob_nonrot_dtor_xyz),
              TCAST(dihedral_offset_for_res),
              TCAST(dihedral_atom_inds),
              TCAST(rotres2resid),
              TCAST(rotameric_chi_desc),
              TCAST(semirotameric_chi_desc));

          dT_dI = result.tensor;
        }));

    return {
        dT_dI,           torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
        torch::Tensor(), torch::Tensor(), torch::Tensor(),
    };
  }
};

template <template <tmol::Device> class DispatchMethod>
class DunbrackPoseScoreOp
    : public torch::autograd::Function<DunbrackPoseScoreOp<DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
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

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "dunbrack_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              DunbrackPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::
                  forward(
                      TCAST(coords),
                      TCAST(pose_stack_block_coord_offset),
                      TCAST(pose_stack_block_type),
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
          coords.type(), "dunbrack_pose_score_backward", ([&] {
            using Real = scalar_t;
            constexpr tmol::Device Dev = device_t;

            auto result = DunbrackPoseScoreDispatch<
                common::DeviceOperations,
                Dev,
                Real,
                Int>::
                backward(
                    TCAST(coords),
                    TCAST(pose_stack_block_coord_offset),
                    TCAST(pose_stack_block_type),
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

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor()};
  }
};

template <template <tmol::Device> class DispatchMethod>
Tensor dunbrack_pose_scores_op(
    Tensor coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
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
      coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
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
Tensor dun_op(
    Tensor coords,
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
    Tensor semirotameric_tables,
    Tensor semirot_table_sizes,
    Tensor semirot_table_strides,
    Tensor semirot_start,
    Tensor semirot_step,
    Tensor semirot_periodicity,
    Tensor rotameric_rotind2tableind,
    Tensor semirotameric_rotind2tableind,
    Tensor ndihe_for_res,
    Tensor dihedral_offset_for_res,
    Tensor dihedral_atom_inds,
    Tensor rottable_set_for_res,
    Tensor nchi_for_res,
    Tensor nrotameric_chi_for_res,
    Tensor rotres2resid,
    Tensor prob_table_offset_for_rotresidue,
    Tensor rotind2tableind_offset_for_res,
    Tensor rotmean_table_offset_for_residue,
    Tensor rotameric_chi_desc,
    Tensor semirotameric_chi_desc,
    Tensor dihedrals,
    Tensor ddihe_dxyz,
    Tensor rotameric_rottable_assignment,
    Tensor semirotameric_rottable_assignment) {
  return DunOp<ScoreDispatch, DispatchMethod>::apply(
      coords,
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
      semirotameric_tables,
      semirot_table_sizes,
      semirot_table_strides,
      semirot_start,
      semirot_step,
      semirot_periodicity,
      rotameric_rotind2tableind,
      semirotameric_rotind2tableind,
      ndihe_for_res,
      dihedral_offset_for_res,
      dihedral_atom_inds,
      rottable_set_for_res,
      nchi_for_res,
      nrotameric_chi_for_res,
      rotres2resid,
      prob_table_offset_for_rotresidue,
      rotind2tableind_offset_for_res,
      rotmean_table_offset_for_residue,
      rotameric_chi_desc,
      semirotameric_chi_desc,
      dihedrals,
      ddihe_dxyz,
      rotameric_rottable_assignment,
      semirotameric_rottable_assignment);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_dun", &dun_op<DunbrackDispatch, common::ForallDispatch>);
  m.def("dunbrack_pose_scores", &dunbrack_pose_scores_op<DeviceOperations>);
}

}  // namespace potentials
}  // namespace dunbrack
}  // namespace score
}  // namespace tmol
