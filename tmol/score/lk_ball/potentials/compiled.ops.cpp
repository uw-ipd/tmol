#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "gen_pose_waters.hh"
// #include "rotamer_pair_energy_lkball.hh"
#include "lk_ball_pose_score.hh"

namespace tmol {
namespace score {
namespace lk_ball {
namespace potentials {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class PoseWaterGen : public torch::autograd::Function<PoseWaterGen> {
 public:
  static Tensor forward(
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
      Tensor pose_stack_inter_residue_connections,
      Tensor block_type_n_atoms,

      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,
      Tensor block_type_n_all_bonds,
      Tensor block_type_all_bonds,
      Tensor block_type_atom_all_bond_ranges,

      Tensor block_type_tile_n_donH,
      Tensor block_type_tile_n_acc,
      Tensor block_type_tile_donH_inds,
      Tensor block_type_tile_don_hvy_inds,
      Tensor block_type_tile_which_donH_for_hvy,

      Tensor block_type_tile_acc_inds,
      Tensor block_type_tile_hybridization,
      Tensor block_type_tile_acc_n_attached_H,
      Tensor block_type_atom_is_hydrogen,
      Tensor global_params,

      Tensor sp2_water_tors,
      Tensor sp3_water_tors,
      Tensor ring_water_tors) {
    at::Tensor waters;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "watergen_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GeneratePoseWaters<common::DeviceOperations, Dev, Real, Int>::
                  forward(
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

                      TCAST(pose_stack_inter_residue_connections),
                      TCAST(block_type_n_atoms),

                      TCAST(block_type_n_interblock_bonds),
                      TCAST(block_type_atoms_forming_chemical_bonds),
                      TCAST(block_type_n_all_bonds),
                      TCAST(block_type_all_bonds),
                      TCAST(block_type_atom_all_bond_ranges),

                      TCAST(block_type_tile_n_donH),
                      TCAST(block_type_tile_n_acc),
                      TCAST(block_type_tile_donH_inds),
                      TCAST(block_type_tile_don_hvy_inds),
                      TCAST(block_type_tile_which_donH_for_hvy),

                      TCAST(block_type_tile_acc_inds),
                      TCAST(block_type_tile_hybridization),
                      TCAST(block_type_tile_acc_n_attached_H),
                      TCAST(block_type_atom_is_hydrogen),
                      TCAST(global_params),

                      TCAST(sp2_water_tors),
                      TCAST(sp3_water_tors),
                      TCAST(ring_water_tors));

          waters = result.tensor;
        }));

    auto max_n_rots_per_pose_tp =
        TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);
    ctx->save_for_backward(
        {// common params
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
         max_n_rots_per_pose_tp.tensor,

         pose_stack_inter_residue_connections,
         block_type_n_atoms,

         block_type_n_interblock_bonds,
         block_type_atoms_forming_chemical_bonds,
         block_type_n_all_bonds,
         block_type_all_bonds,
         block_type_atom_all_bond_ranges,

         block_type_tile_n_donH,
         block_type_tile_n_acc,
         block_type_tile_donH_inds,
         block_type_tile_don_hvy_inds,
         block_type_tile_which_donH_for_hvy,

         block_type_tile_acc_inds,
         block_type_tile_hybridization,
         block_type_tile_acc_n_attached_H,
         block_type_atom_is_hydrogen,
         global_params,

         sp2_water_tors,
         sp3_water_tors,
         ring_water_tors});

    return waters;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

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

    auto pose_stack_inter_residue_connections = saved[i++];
    auto block_type_n_atoms = saved[i++];

    auto block_type_n_interblock_bonds = saved[i++];
    auto block_type_atoms_forming_chemical_bonds = saved[i++];
    auto block_type_n_all_bonds = saved[i++];
    auto block_type_all_bonds = saved[i++];
    auto block_type_atom_all_bond_ranges = saved[i++];

    auto block_type_tile_n_donH = saved[i++];
    auto block_type_tile_n_acc = saved[i++];
    auto block_type_tile_donH_inds = saved[i++];
    auto block_type_tile_don_hvy_inds = saved[i++];
    auto block_type_tile_which_donH_for_hvy = saved[i++];

    auto block_type_tile_acc_inds = saved[i++];
    auto block_type_tile_hybridization = saved[i++];
    auto block_type_tile_acc_n_attached_H = saved[i++];
    auto block_type_atom_is_hydrogen = saved[i++];
    auto global_params = saved[i++];

    auto sp2_water_tors = saved[i++];
    auto sp3_water_tors = saved[i++];
    auto ring_water_tors = saved[i++];

    at::Tensor dT_d_pose_coords;

    using Int = int32_t;

    auto dE_dWxyz = grad_outputs[0];

    // for (int i = 0; i < dE_dWxyz.size(0); i++)
    //   for (int j = 0; j < dE_dWxyz.size(1); j++)
    //     printf(
    //         "DE_WXYZ: %i %i %f %f %f\n",
    //         i,
    //         j,
    //         dE_dWxyz[i][j][0].item<float>(),
    //         dE_dWxyz[i][j][1].item<float>(),
    //         dE_dWxyz[i][j][2].item<float>());

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "WaterGenOpBackward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GeneratePoseWaters<common::DeviceOperations, Dev, Real, Int>::
                  backward(
                      TCAST(dE_dWxyz),
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
                      TCAST(pose_stack_inter_residue_connections),
                      TCAST(block_type_n_atoms),
                      TCAST(block_type_n_interblock_bonds),
                      TCAST(block_type_atoms_forming_chemical_bonds),
                      TCAST(block_type_n_all_bonds),
                      TCAST(block_type_all_bonds),
                      TCAST(block_type_atom_all_bond_ranges),
                      TCAST(block_type_tile_n_donH),
                      TCAST(block_type_tile_n_acc),
                      TCAST(block_type_tile_donH_inds),
                      TCAST(block_type_tile_don_hvy_inds),
                      TCAST(block_type_tile_which_donH_for_hvy),
                      TCAST(block_type_tile_acc_inds),
                      TCAST(block_type_tile_hybridization),
                      TCAST(block_type_tile_acc_n_attached_H),
                      TCAST(block_type_atom_is_hydrogen),
                      TCAST(global_params),
                      TCAST(sp2_water_tors),
                      TCAST(sp3_water_tors),
                      TCAST(ring_water_tors));
          dT_d_pose_coords = result.tensor;
        }));

    return {dT_d_pose_coords, torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(), torch::Tensor()};
  };
};

Tensor pose_watergen_op(
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

    // term specific params
    Tensor pose_stack_inter_residue_connections,

    Tensor block_type_n_atoms,

    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,
    Tensor block_type_n_all_bonds,
    Tensor block_type_all_bonds,
    Tensor block_type_atom_all_bond_ranges,

    Tensor block_type_tile_n_donH,
    Tensor block_type_tile_n_acc,
    Tensor block_type_tile_donH_inds,
    Tensor block_type_tile_don_hvy_inds,
    Tensor block_type_tile_which_donH_for_hvy,

    Tensor block_type_tile_acc_inds,
    Tensor block_type_tile_hybridization,
    Tensor block_type_tile_acc_n_attached_H,
    Tensor block_type_atom_is_hydrogen,
    Tensor global_params,

    Tensor sp2_water_tors,
    Tensor sp3_water_tors,
    Tensor ring_water_tors) {
  return PoseWaterGen::apply(
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
      pose_stack_inter_residue_connections,
      block_type_n_atoms,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,
      block_type_n_all_bonds,
      block_type_all_bonds,
      block_type_atom_all_bond_ranges,
      block_type_tile_n_donH,
      block_type_tile_n_acc,
      block_type_tile_donH_inds,
      block_type_tile_don_hvy_inds,
      block_type_tile_which_donH_for_hvy,
      block_type_tile_acc_inds,
      block_type_tile_hybridization,
      block_type_tile_acc_n_attached_H,
      block_type_atom_is_hydrogen,
      global_params,
      sp2_water_tors,
      sp3_water_tors,
      ring_water_tors);
};

// template <template <tmol::Device> class Dispatch>
// Tensor rotamer_pair_energies(
//     Tensor context_coords,
//     Tensor context_block_type,
//     Tensor alternate_coords,
//     Tensor alternate_ids,

//     Tensor context_water_coords,

//     Tensor context_system_ids,
//     Tensor system_min_bond_separation,
//     Tensor system_inter_block_bondsep,
//     Tensor system_neighbor_list,

//     // parameters to build waters
//     Tensor bt_is_acceptor,
//     Tensor bt_acceptor_type,
//     Tensor bt_acceptor_hybridization,
//     Tensor bt_acceptor_base_ind,

//     Tensor bt_is_donor,
//     Tensor bt_donor_type,
//     Tensor bt_donor_attached_hydrogens,

//     // Tensor lkb_water_gen_type_params,
//     Tensor lkb_global_params,
//     Tensor sp2_water_tors,
//     Tensor sp3_water_tors,
//     Tensor ring_water_tors,

//     Tensor lkb_weight

// ) {
//   at::Tensor rpes;
//   at::Tensor event;

//   using Int = int32_t;
//   constexpr int MAX_WATER = 4;

//   TMOL_DISPATCH_FLOATING_DEVICE(
//       context_coords.options(), "rotamer_rpe_evaluation", ([&] {
//         using Real = scalar_t;
//         constexpr tmol::Device Dev = device_t;

//         auto result = LKBallRPEDispatch<Dispatch, Dev, Real, Int,
//         MAX_WATER>::f(
//             TCAST(context_coords),
//             TCAST(context_block_type),
//             TCAST(alternate_coords),
//             TCAST(alternate_ids),

//             TCAST(context_water_coords),

//             TCAST(context_system_ids),
//             TCAST(system_min_bond_separation),
//             TCAST(system_inter_block_bondsep),
//             TCAST(system_neighbor_list),

//             TCAST(bt_is_acceptor),
//             TCAST(bt_acceptor_type),
//             TCAST(bt_acceptor_hybridization),
//             TCAST(bt_acceptor_base_ind),

//             TCAST(bt_is_donor),
//             TCAST(bt_donor_type),
//             TCAST(bt_donor_attached_hydrogens),

//             TCAST(lkb_global_params),
//             TCAST(sp2_water_tors),
//             TCAST(sp3_water_tors),
//             TCAST(ring_water_tors));
//         rpes = std::get<0>(result).tensor;
//       }));
//   return rpes;
// }

class LKBallPoseScoreOp : public torch::autograd::Function<LKBallPoseScoreOp> {
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
      Tensor pose_stack_inter_residue_connections,

      Tensor pose_stack_min_bond_separation,
      Tensor pose_stack_inter_block_bondsep,
      Tensor block_type_n_atoms,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,

      Tensor block_type_tile_n_polar_atoms,
      Tensor block_type_tile_n_occluder_atoms,
      Tensor block_type_tile_pol_occ_inds,
      Tensor block_type_tile_lk_ball_params,
      Tensor block_type_path_distance,

      Tensor global_params,
      Tensor water_coords,
      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "lk_ball_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallPoseScoreDispatch<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              forward(
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
                  TCAST(pose_stack_inter_residue_connections),

                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),

                  TCAST(block_type_tile_n_polar_atoms),
                  TCAST(block_type_tile_n_occluder_atoms),
                  TCAST(block_type_tile_pol_occ_inds),
                  TCAST(block_type_tile_lk_ball_params),
                  TCAST(block_type_path_distance),

                  TCAST(global_params),
                  TCAST(water_coords),
                  output_block_pair_energies);

          score = std::get<0>(result).tensor;
          block_neighbors = std::get<1>(result).tensor;
          // std::cout << "Shape of score with " << score.dim() << " dimensions"
          // << std::endl; for (int i = 0; i < score.dim(); ++i) {
          //   std::cout << score.size(i) << " ";
          // }
          // std::cout << std::endl;
          // std::cout << "Scores " << std::endl;
          // for (int j = 0; j < score.size(1); ++j) {
          //     printf("score %2d [%8.4f %8.4f %8.4f %8.4f]\n",
          //       j,
          //       score[0][j][0][0].item<float>(),
          //       score[1][j][0][0].item<float>(),
          //       score[2][j][0][0].item<float>(),
          //       score[3][j][0][0].item<float>());
          // }

          // std::cout << "Shape of block_neighbors with " <<
          // block_neighbors.dim() << " dimensions" << std::endl; for (int i =
          // 0; i < block_neighbors.dim(); ++i) {
          //   std::cout << block_neighbors.size(i) << " ";
          // }
          // std::cout << std::endl;
        }));

    if (!output_block_pair_energies) {
      score = score.squeeze(-1).squeeze(-1);
    }

    auto max_n_rots_per_pose_tp =
        TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);

    ctx->save_for_backward(
        {// common params
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
         max_n_rots_per_pose_tp.tensor,
         pose_stack_inter_residue_connections,

         pose_stack_min_bond_separation,
         pose_stack_inter_block_bondsep,
         block_type_n_atoms,
         block_type_n_interblock_bonds,
         block_type_atoms_forming_chemical_bonds,

         block_type_tile_n_polar_atoms,
         block_type_tile_n_occluder_atoms,
         block_type_tile_pol_occ_inds,
         block_type_tile_lk_ball_params,
         block_type_path_distance,

         global_params,
         water_coords,
         block_neighbors});

    ctx->saved_data["block_pair_scoring"] = output_block_pair_energies;

    // std::cout << "Leaving LKBallPoseScoreOp forward" << std::endl;
    return {score, block_neighbors};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    // std::cout << "Entering LKBallPoseScoreOp backward" << std::endl;
    auto saved = ctx->get_saved_variables();

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
    auto pose_stack_inter_residue_connections = saved[i++];

    auto pose_stack_min_bond_separation = saved[i++];
    auto pose_stack_inter_block_bondsep = saved[i++];
    auto block_type_n_atoms = saved[i++];
    auto block_type_n_interblock_bonds = saved[i++];
    auto block_type_atoms_forming_chemical_bonds = saved[i++];

    auto block_type_tile_n_polar_atoms = saved[i++];
    auto block_type_tile_n_occluder_atoms = saved[i++];
    auto block_type_tile_pol_occ_inds = saved[i++];
    auto block_type_tile_lk_ball_params = saved[i++];
    auto block_type_path_distance = saved[i++];

    auto global_params = saved[i++];
    auto water_coords = saved[i++];
    auto block_neighbors = saved[i++];

    at::Tensor dV_d_pose_coords, dV_d_water_coords;
    using Int = int32_t;

    auto dTdV = grad_outputs[0];

    bool block_pair_scoring = ctx->saved_data["block_pair_scoring"].toBool();
    if (!block_pair_scoring) {
      dTdV = dTdV.unsqueeze(-1).unsqueeze(-1);
    }

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "lk_ball_pose_score_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallPoseScoreDispatch<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              backward(
                  TCAST(rot_coords),
                  TCAST(water_coords),
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
                  TCAST(pose_stack_inter_residue_connections),

                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),

                  TCAST(block_type_tile_n_polar_atoms),
                  TCAST(block_type_tile_n_occluder_atoms),
                  TCAST(block_type_tile_pol_occ_inds),
                  TCAST(block_type_tile_lk_ball_params),
                  TCAST(block_type_path_distance),

                  TCAST(global_params),
                  TCAST(block_neighbors),
                  TCAST(dTdV),
                  block_pair_scoring);

          dV_d_pose_coords = std::get<0>(result).tensor;
          dV_d_water_coords = std::get<1>(result).tensor;
        }));

    // std::cout << "Leaving LKBallPoseScoreOp backward" << std::endl;
    return {
        dV_d_pose_coords, torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  dV_d_water_coords, torch::Tensor(),

    };
  }
};

class LKBallRotamerScoreOp
    : public torch::autograd::Function<LKBallRotamerScoreOp> {
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
      Tensor pose_stack_inter_residue_connections,

      Tensor pose_stack_min_bond_separation,
      Tensor pose_stack_inter_block_bondsep,
      Tensor block_type_n_atoms,
      Tensor block_type_n_interblock_bonds,
      Tensor block_type_atoms_forming_chemical_bonds,

      Tensor block_type_tile_n_polar_atoms,
      Tensor block_type_tile_n_occluder_atoms,
      Tensor block_type_tile_pol_occ_inds,
      Tensor block_type_tile_lk_ball_params,
      Tensor block_type_path_distance,

      Tensor global_params,
      Tensor water_coords,
      bool output_block_pair_energies) {
    at::Tensor score;
    at::Tensor dispatch_indices;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "lk_ball_rotamer_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallRotamerScoreDispatch<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              forward(
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
                  TCAST(pose_stack_inter_residue_connections),

                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),

                  TCAST(block_type_tile_n_polar_atoms),
                  TCAST(block_type_tile_n_occluder_atoms),
                  TCAST(block_type_tile_pol_occ_inds),
                  TCAST(block_type_tile_lk_ball_params),
                  TCAST(block_type_path_distance),

                  TCAST(global_params),
                  TCAST(water_coords),
                  output_block_pair_energies);

          score = std::get<0>(result).tensor;
          dispatch_indices = std::get<1>(result).tensor;
        }));

    auto max_n_rots_per_pose_tp =
        TPack<Int, 1, tmol::Device::CPU>::full(1, max_n_rots_per_pose);
    ctx->save_for_backward(
        {// common params
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
         max_n_rots_per_pose_tp.tensor,
         pose_stack_inter_residue_connections,

         pose_stack_min_bond_separation,
         pose_stack_inter_block_bondsep,
         block_type_n_atoms,
         block_type_n_interblock_bonds,
         block_type_atoms_forming_chemical_bonds,

         block_type_tile_n_polar_atoms,
         block_type_tile_n_occluder_atoms,
         block_type_tile_pol_occ_inds,
         block_type_tile_lk_ball_params,
         block_type_path_distance,

         global_params,
         water_coords,
         dispatch_indices});

    ctx->saved_data["block_pair_scoring"] = output_block_pair_energies;

    return {score, dispatch_indices};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

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
    auto pose_stack_inter_residue_connections = saved[i++];

    auto pose_stack_min_bond_separation = saved[i++];
    auto pose_stack_inter_block_bondsep = saved[i++];
    auto block_type_n_atoms = saved[i++];
    auto block_type_n_interblock_bonds = saved[i++];
    auto block_type_atoms_forming_chemical_bonds = saved[i++];

    auto block_type_tile_n_polar_atoms = saved[i++];
    auto block_type_tile_n_occluder_atoms = saved[i++];
    auto block_type_tile_pol_occ_inds = saved[i++];
    auto block_type_tile_lk_ball_params = saved[i++];
    auto block_type_path_distance = saved[i++];

    auto global_params = saved[i++];
    auto water_coords = saved[i++];
    auto dispatch_indices = saved[i++];

    at::Tensor dV_d_pose_coords, dV_d_water_coords;
    using Int = int32_t;

    auto dTdV = grad_outputs[0];

    bool block_pair_scoring = ctx->saved_data["block_pair_scoring"].toBool();
    /*if (!block_pair_scoring) {
      dTdV = dTdV.unsqueeze(-1).unsqueeze(-1);
    }*/

    TMOL_DISPATCH_FLOATING_DEVICE(
        rot_coords.options(), "lk_ball_rotamer_score_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallRotamerScoreDispatch<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              backward(
                  TCAST(rot_coords),
                  TCAST(water_coords),
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
                  TCAST(pose_stack_inter_residue_connections),

                  TCAST(pose_stack_min_bond_separation),
                  TCAST(pose_stack_inter_block_bondsep),
                  TCAST(block_type_n_atoms),
                  TCAST(block_type_n_interblock_bonds),
                  TCAST(block_type_atoms_forming_chemical_bonds),

                  TCAST(block_type_tile_n_polar_atoms),
                  TCAST(block_type_tile_n_occluder_atoms),
                  TCAST(block_type_tile_pol_occ_inds),
                  TCAST(block_type_tile_lk_ball_params),
                  TCAST(block_type_path_distance),

                  TCAST(global_params),
                  TCAST(dispatch_indices),
                  TCAST(dTdV),
                  block_pair_scoring);

          dV_d_pose_coords = std::get<0>(result).tensor;
          dV_d_water_coords = std::get<1>(result).tensor;
        }));

    return {
        dV_d_pose_coords, torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  torch::Tensor(),   torch::Tensor(), torch::Tensor(),
        torch::Tensor(),  dV_d_water_coords, torch::Tensor(),

    };
  }
};

std::vector<Tensor> lkball_pose_score(
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

    Tensor pose_stack_inter_residue_connections,

    Tensor pose_stack_min_bond_separation,
    Tensor pose_stack_inter_block_bondsep,
    Tensor block_type_n_atoms,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,

    Tensor block_type_tile_n_polar_atoms,
    Tensor block_type_tile_n_occluder_atoms,
    Tensor block_type_tile_pol_occ_inds,
    Tensor block_type_tile_lk_ball_params,
    Tensor block_type_path_distance,

    Tensor global_params,
    Tensor water_coords,
    bool output_block_pair_energies) {
  return LKBallPoseScoreOp::apply(
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
      pose_stack_inter_residue_connections,

      pose_stack_min_bond_separation,
      pose_stack_inter_block_bondsep,
      block_type_n_atoms,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,

      block_type_tile_n_polar_atoms,
      block_type_tile_n_occluder_atoms,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      block_type_path_distance,

      global_params,
      water_coords,
      output_block_pair_energies);
}

std::vector<Tensor> lkball_rotamer_score(
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

    Tensor pose_stack_inter_residue_connections,

    Tensor pose_stack_min_bond_separation,
    Tensor pose_stack_inter_block_bondsep,
    Tensor block_type_n_atoms,
    Tensor block_type_n_interblock_bonds,
    Tensor block_type_atoms_forming_chemical_bonds,

    Tensor block_type_tile_n_polar_atoms,
    Tensor block_type_tile_n_occluder_atoms,
    Tensor block_type_tile_pol_occ_inds,
    Tensor block_type_tile_lk_ball_params,
    Tensor block_type_path_distance,

    Tensor global_params,
    Tensor water_coords,
    bool output_block_pair_energies) {
  return LKBallRotamerScoreOp::apply(
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
      pose_stack_inter_residue_connections,

      pose_stack_min_bond_separation,
      pose_stack_inter_block_bondsep,
      block_type_n_atoms,
      block_type_n_interblock_bonds,
      block_type_atoms_forming_chemical_bonds,

      block_type_tile_n_polar_atoms,
      block_type_tile_n_occluder_atoms,
      block_type_tile_pol_occ_inds,
      block_type_tile_lk_ball_params,
      block_type_path_distance,

      global_params,
      water_coords,
      output_block_pair_energies);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  // m.def(
  //     "score_lkball_inter_system_scores",
  //     &rotamer_pair_energies<common::ForallDispatch>);
  m.def("lk_ball_pose_score", &lkball_pose_score);
  m.def("lk_ball_rotamer_score", &lkball_rotamer_score);
  m.def("gen_pose_waters", &pose_watergen_op);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
