#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "dispatch.hh"
#include "gen_waters.hh"
#include "gen_pose_waters.hh"
#include "rotamer_pair_energy_lkball.hh"
#include "lk_ball_pose_score.hh"
#include "lk_ball_pose_score2.hh"

namespace tmol {
namespace score {
namespace lk_ball {
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
      Tensor I,
      Tensor polars_I,
      Tensor atom_type_I,
      Tensor waters_I,
      Tensor J,
      Tensor occluders_J,
      Tensor atom_type_J,
      Tensor waters_J,
      Tensor bonded_path_lengths,
      Tensor type_params,
      Tensor global_params) {
    at::Tensor score;

    using Int = int64_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        I.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::forward(
              TCAST(I),
              TCAST(polars_I),
              TCAST(atom_type_I),
              TCAST(waters_I),
              TCAST(J),
              TCAST(occluders_J),
              TCAST(atom_type_J),
              TCAST(waters_J),
              TCAST(bonded_path_lengths),
              TCAST(type_params),
              TCAST(global_params));

          score = result.tensor;
        }));

    ctx->save_for_backward({I,
                            polars_I,
                            atom_type_I,
                            waters_I,
                            J,
                            occluders_J,
                            atom_type_J,
                            waters_J,
                            bonded_path_lengths,
                            type_params,
                            global_params});

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;
    auto I = saved[i++];
    auto polars_I = saved[i++];
    auto atom_type_I = saved[i++];
    auto waters_I = saved[i++];
    auto J = saved[i++];
    auto occluders_J = saved[i++];
    auto atom_type_J = saved[i++];
    auto waters_J = saved[i++];
    auto bonded_path_lengths = saved[i++];
    auto type_params = saved[i++];
    auto global_params = saved[i++];

    at::Tensor dV_dI, dV_dJ, dV_dwaters_I, dV_dwaters_J;
    using Int = int64_t;

    auto dTdV = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        I.type(), "ScoreOpBackward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::backward(
              TCAST(dTdV),
              TCAST(I),
              TCAST(polars_I),
              TCAST(atom_type_I),
              TCAST(waters_I),
              TCAST(J),
              TCAST(occluders_J),
              TCAST(atom_type_J),
              TCAST(waters_J),
              TCAST(bonded_path_lengths),
              TCAST(type_params),
              TCAST(global_params));

          dV_dI = std::get<0>(result).tensor;
          dV_dJ = std::get<1>(result).tensor;
          dV_dwaters_I = std::get<2>(result).tensor;
          dV_dwaters_J = std::get<3>(result).tensor;
        }));

    return {dV_dI,
            torch::Tensor(),
            torch::Tensor(),
            dV_dwaters_I,
            dV_dJ,
            torch::Tensor(),
            torch::Tensor(),
            dV_dwaters_J,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  }
};

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int,
        int MAX_WATER>
    class WaterGenDispatch,
    template <tmol::Device>
    class DispatchMethod>
class WaterGen : public Function<WaterGen<WaterGenDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor atom_types,
      Tensor indexed_bonds,
      Tensor indexed_bond_spans,
      Tensor type_params,
      Tensor global_params,
      Tensor sp2_water_tors,
      Tensor sp3_water_tors,
      Tensor ring_water_tors) {
    at::Tensor waters;

    using Int = int64_t;
    constexpr int MAX_WATER = 4;

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "watergen_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              WaterGenDispatch<DispatchMethod, Dev, Real, Int, MAX_WATER>::
                  forward(
                      TCAST(coords),
                      TCAST(atom_types),
                      TCAST(indexed_bonds),
                      TCAST(indexed_bond_spans),
                      TCAST(type_params),
                      TCAST(global_params),
                      TCAST(sp2_water_tors),
                      TCAST(sp3_water_tors),
                      TCAST(ring_water_tors));

          waters = result.tensor;
        }));

    ctx->save_for_backward({coords,
                            atom_types,
                            indexed_bonds,
                            indexed_bond_spans,
                            type_params,
                            global_params,
                            sp2_water_tors,
                            sp3_water_tors,
                            ring_water_tors});

    return waters;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;

    auto coords = saved[i++];
    auto atom_types = saved[i++];
    auto indexed_bonds = saved[i++];
    auto indexed_bond_spans = saved[i++];
    auto type_params = saved[i++];
    auto global_params = saved[i++];
    auto sp2_water_tors = saved[i++];
    auto sp3_water_tors = saved[i++];
    auto ring_water_tors = saved[i++];

    at::Tensor dT_d_coords;
    using Int = int64_t;

    constexpr int MAX_WATER = 4;
    auto dT_d_waters = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        coords.type(), "WaterGenOpBackward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              WaterGenDispatch<DispatchMethod, Dev, Real, Int, MAX_WATER>::
                  backward(
                      TCAST(dT_d_waters),
                      TCAST(coords),
                      TCAST(atom_types),
                      TCAST(indexed_bonds),
                      TCAST(indexed_bond_spans),
                      TCAST(type_params),
                      TCAST(global_params),
                      TCAST(sp2_water_tors),
                      TCAST(sp3_water_tors),
                      TCAST(ring_water_tors));

          dT_d_coords = result.tensor;
        }));

    return {dT_d_coords,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()};
  };
};

class PoseWaterGen : public torch::autograd::Function<PoseWaterGen> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor pose_coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
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
        pose_coords.type(), "watergen_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GeneratePoseWaters<common::DeviceOperations, Dev, Real, Int>::
                  forward(
                      TCAST(pose_coords),
                      TCAST(pose_stack_block_coord_offset),
                      TCAST(pose_stack_block_type),
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

    ctx->save_for_backward({pose_coords,
                            pose_stack_block_coord_offset,
                            pose_stack_block_type,
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

    auto pose_coords = saved[i++];
    auto pose_stack_block_coord_offset = saved[i++];
    auto pose_stack_block_type = saved[i++];
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

    TMOL_DISPATCH_FLOATING_DEVICE(
        pose_coords.type(), "WaterGenOpBackward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result =
              GeneratePoseWaters<common::DeviceOperations, Dev, Real, Int>::
                  backward(
                      TCAST(dE_dWxyz),
                      TCAST(pose_coords),
                      TCAST(pose_stack_block_coord_offset),
                      TCAST(pose_stack_block_type),
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
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor(),
            torch::Tensor(),  torch::Tensor(),

            torch::Tensor(),  torch::Tensor(), torch::Tensor()};
  };
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
    Tensor I,
    Tensor polars_I,
    Tensor atom_type_I,
    Tensor waters_I,
    Tensor J,
    Tensor occluders_J,
    Tensor atom_type_J,
    Tensor waters_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  return ScoreOp<ScoreDispatch, DispatchMethod>::apply(
      I,
      polars_I,
      atom_type_I,
      waters_I,
      J,
      occluders_J,
      atom_type_J,
      waters_J,
      bonded_path_lengths,
      type_params,
      global_params);
}

template <
    template <
        template <tmol::Device>
        class Dispatch,
        tmol::Device D,
        typename Real,
        typename Int,
        int MAX_WATER>
    class WaterGenDispatch,
    template <tmol::Device>
    class DispatchMethod>
Tensor watergen_op(
    Tensor coords,
    Tensor atom_types,
    Tensor indexed_bonds,
    Tensor indexed_bond_spans,
    Tensor type_params,
    Tensor global_params,
    Tensor sp2_water_tors,
    Tensor sp3_water_tors,
    Tensor ring_water_tors) {
  return WaterGen<WaterGenDispatch, DispatchMethod>::apply(
      coords,
      atom_types,
      indexed_bonds,
      indexed_bond_spans,
      type_params,
      global_params,
      sp2_water_tors,
      sp3_water_tors,
      ring_water_tors);
};

Tensor pose_watergen_op(
    Tensor pose_coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
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
      pose_coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
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

template <template <tmol::Device> class Dispatch>
Tensor rotamer_pair_energies(
    Tensor context_coords,
    Tensor context_block_type,
    Tensor alternate_coords,
    Tensor alternate_ids,

    Tensor context_water_coords,

    Tensor context_system_ids,
    Tensor system_min_bond_separation,
    Tensor system_inter_block_bondsep,
    Tensor system_neighbor_list,

    // parameters to build waters
    Tensor bt_is_acceptor,
    Tensor bt_acceptor_type,
    Tensor bt_acceptor_hybridization,
    Tensor bt_acceptor_base_ind,

    Tensor bt_is_donor,
    Tensor bt_donor_type,
    Tensor bt_donor_attached_hydrogens,

    // Tensor lkb_water_gen_type_params,
    Tensor lkb_global_params,
    Tensor sp2_water_tors,
    Tensor sp3_water_tors,
    Tensor ring_water_tors,

    Tensor lkb_weight

) {
  at::Tensor rpes;
  at::Tensor event;

  using Int = int32_t;
  constexpr int MAX_WATER = 4;

  TMOL_DISPATCH_FLOATING_DEVICE(
      context_coords.type(), "rotamer_rpe_evaluation", ([&] {
        using Real = scalar_t;
        constexpr tmol::Device Dev = device_t;

        auto result = LKBallRPEDispatch<Dispatch, Dev, Real, Int, MAX_WATER>::f(
            TCAST(context_coords),
            TCAST(context_block_type),
            TCAST(alternate_coords),
            TCAST(alternate_ids),

            TCAST(context_water_coords),

            TCAST(context_system_ids),
            TCAST(system_min_bond_separation),
            TCAST(system_inter_block_bondsep),
            TCAST(system_neighbor_list),

            TCAST(bt_is_acceptor),
            TCAST(bt_acceptor_type),
            TCAST(bt_acceptor_hybridization),
            TCAST(bt_acceptor_base_ind),

            TCAST(bt_is_donor),
            TCAST(bt_donor_type),
            TCAST(bt_donor_attached_hydrogens),

            TCAST(lkb_global_params),
            TCAST(sp2_water_tors),
            TCAST(sp3_water_tors),
            TCAST(ring_water_tors));
        rpes = std::get<0>(result).tensor;
      }));
  return rpes;
}

class LKBallPoseScoreOp : public torch::autograd::Function<LKBallPoseScoreOp> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor pose_coords,
      Tensor water_coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
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

      Tensor global_params) {
    at::Tensor score;
    at::Tensor block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        pose_coords.type(), "lk_ball_pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallPoseScoreDispatch<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              forward(
                  TCAST(pose_coords),
                  TCAST(water_coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
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

                  TCAST(global_params));

          score = std::get<0>(result).tensor;
          block_neighbors = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({pose_coords,
                            water_coords,
                            pose_stack_block_coord_offset,
                            pose_stack_block_type,
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
                            block_neighbors});

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;

    auto pose_coords = saved[i++];
    auto water_coords = saved[i++];
    auto pose_stack_block_coord_offset = saved[i++];
    auto pose_stack_block_type = saved[i++];
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
    auto block_neighbors = saved[i++];

    at::Tensor dV_d_pose_coords, dV_d_water_coords;
    using Int = int32_t;

    auto dTdV = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        pose_coords.type(), "lk_ball_pose_score_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallPoseScoreDispatch<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              backward(
                  TCAST(pose_coords),
                  TCAST(water_coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
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
                  TCAST(dTdV));

          dV_d_pose_coords = std::get<0>(result).tensor;
          dV_d_water_coords = std::get<1>(result).tensor;
        }));

    return {dV_d_pose_coords,
            dV_d_water_coords,
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

            torch::Tensor()};
  }
};

Tensor lkball_pose_score(
    Tensor pose_coords,
    Tensor water_coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
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

    Tensor global_params) {
  return LKBallPoseScoreOp::apply(
      pose_coords,
      water_coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
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

      global_params);
}

class LKBallPoseScoreOp2
    : public torch::autograd::Function<LKBallPoseScoreOp2> {
 public:
  static Tensor forward(
      AutogradContext* ctx,

      Tensor pose_coords,
      Tensor water_coords,
      Tensor pose_stack_block_coord_offset,
      Tensor pose_stack_block_type,
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

      Tensor global_params) {
    at::Tensor score;
    at::Tensor block_neighbors;

    using Int = int32_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        pose_coords.type(), "lk_ball_pose_score_op2", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallPoseScoreDispatch2<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              forward(
                  TCAST(pose_coords),
                  TCAST(water_coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
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

                  TCAST(global_params));

          score = std::get<0>(result).tensor;
          block_neighbors = std::get<1>(result).tensor;
        }));

    ctx->save_for_backward({pose_coords,
                            water_coords,
                            pose_stack_block_coord_offset,
                            pose_stack_block_type,
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
                            block_neighbors});

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    int i = 0;

    auto pose_coords = saved[i++];
    auto water_coords = saved[i++];
    auto pose_stack_block_coord_offset = saved[i++];
    auto pose_stack_block_type = saved[i++];
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
    auto block_neighbors = saved[i++];

    at::Tensor dV_d_pose_coords, dV_d_water_coords;
    using Int = int32_t;

    auto dTdV = grad_outputs[0];

    TMOL_DISPATCH_FLOATING_DEVICE(
        pose_coords.type(), "lk_ball_pose_score_backward", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LKBallPoseScoreDispatch2<
              common::DeviceOperations,
              Dev,
              Real,
              Int>::
              backward(
                  TCAST(pose_coords),
                  TCAST(water_coords),
                  TCAST(pose_stack_block_coord_offset),
                  TCAST(pose_stack_block_type),
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
                  TCAST(dTdV));

          dV_d_pose_coords = std::get<0>(result).tensor;
          dV_d_water_coords = std::get<1>(result).tensor;
        }));

    return {dV_d_pose_coords,
            dV_d_water_coords,
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

            torch::Tensor()};
  }
};

Tensor lkball_pose_score2(
    Tensor pose_coords,
    Tensor water_coords,
    Tensor pose_stack_block_coord_offset,
    Tensor pose_stack_block_type,
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

    Tensor global_params) {
  return LKBallPoseScoreOp2::apply(
      pose_coords,
      water_coords,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
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

      global_params);
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_lkball", &score_op<LKBallDispatch, common::AABBDispatch>);
  m.def(
      "watergen_lkball", &watergen_op<GenerateWaters, common::ForallDispatch>);
  m.def(
      "score_lkball_inter_system_scores",
      &rotamer_pair_energies<common::ForallDispatch>);
  m.def("lk_ball_pose_score", &lkball_pose_score);
  m.def("gen_pose_waters", &pose_watergen_op);
}

}  // namespace potentials
}  // namespace lk_ball
}  // namespace score
}  // namespace tmol
