#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>

#include "lj.dispatch.hh"
#include "lk_isotropic.dispatch.hh"
#include "rotamer_pair_energy_lj.hh"
#include "rotamer_pair_energy_lk.hh"

namespace tmol {
namespace score {
namespace ljlk {
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
class LJScoreOp
    : public torch::autograd::Function<LJScoreOp<ScoreDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor I,
      Tensor atom_type_I,
      Tensor J,
      Tensor atom_type_J,
      Tensor bonded_path_lengths,
      Tensor type_params,
      Tensor global_params) {
    at::Tensor score;
    at::Tensor dScore_dI;
    at::Tensor dScore_dJ;

    using Int = int64_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        I.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
              TCAST(I),
              TCAST(atom_type_I),
              TCAST(J),
              TCAST(atom_type_J),
              TCAST(bonded_path_lengths),
              TCAST(type_params),
              TCAST(global_params));

          score = std::get<0>(result).tensor;
          dScore_dI = std::get<1>(result).tensor;
          dScore_dJ = std::get<2>(result).tensor;
        }));

    ctx->save_for_backward({dScore_dI, dScore_dJ});
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
    auto dI = result[i++];
    auto dJ = result[i++];

    return {
        dI,
        torch::Tensor(),
        dJ,
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
class LKScoreOp
    : public torch::autograd::Function<LKScoreOp<ScoreDispatch, DispatchMethod>> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor I,
      Tensor atom_type_I,
      Tensor heavyatom_inds_I,
      Tensor J,
      Tensor atom_type_J,
      Tensor heavyatom_inds_J,
      Tensor bonded_path_lengths,
      Tensor type_params,
      Tensor global_params) {
    at::Tensor score;
    at::Tensor dScore_dI;
    at::Tensor dScore_dJ;

    using Int = int64_t;

    TMOL_DISPATCH_FLOATING_DEVICE(
        I.type(), "score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
              TCAST(I),
              TCAST(atom_type_I),
              TCAST(heavyatom_inds_I),
              TCAST(J),
              TCAST(atom_type_J),
              TCAST(heavyatom_inds_J),
              TCAST(bonded_path_lengths),
              TCAST(type_params),
              TCAST(global_params));

          score = std::get<0>(result).tensor;
          dScore_dI = std::get<1>(result).tensor;
          dScore_dJ = std::get<2>(result).tensor;
        }));

    ctx->save_for_backward({dScore_dI, dScore_dJ});
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
    auto dI = result[i++];
    auto dJ = result[i++];

    return {
        dI,
        torch::Tensor(),
        torch::Tensor(),
        dJ,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
  }
};


template <
  template <tmol::Device>
  class DispatchMethod>>
class LJLKPoseScoreOp
  : public torch::autograd::Function<LJLKPoseScoreOp<DispatchMethod>> {
public:
  static Tensor forward(
      AutogradContext* ctx,
      Tensor coords,

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
        I.type(), "pose_score_op", ([&] {
          using Real = scalar_t;
          constexpr tmol::Device Dev = device_t;

          auto result = LJLKPoseScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
	    TCAST(coords),

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
	    TCAST(global_params));

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
Tensor lj_score_op(
    Tensor I,
    Tensor atom_type_I,
    Tensor J,
    Tensor atom_type_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  return LJScoreOp<ScoreDispatch, DispatchMethod>::apply(
      I,
      atom_type_I,
      J,
      atom_type_J,
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
        typename Int>
    class ScoreDispatch,
    template <tmol::Device>
    class DispatchMethod>
Tensor lk_score_op(
    Tensor I,
    Tensor atom_type_I,
    Tensor heavyatom_inds_I,
    Tensor J,
    Tensor atom_type_J,
    Tensor heavyatom_inds_J,
    Tensor bonded_path_lengths,
    Tensor type_params,
    Tensor global_params) {
  return LKScoreOp<ScoreDispatch, DispatchMethod>::apply(
      I,
      atom_type_I,
      heavyatom_inds_I,
      J,
      atom_type_J,
      heavyatom_inds_J,
      bonded_path_lengths,
      type_params,
      global_params);
}

template <
  template<tmol::Device>
  class DispatchMethod>
Tensor ljlk_pose_score_op(
  Tensor coords,

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
  Tensor global_params
) {
  return LJLKPoseScoreOp<DispatchMethod>::apply(
    coords,

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


Tensor
rotamer_pair_energies_op(
  Tensor context_coords,
  Tensor context_block_type,
  Tensor alternate_coords,
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
  Tensor lj_lk_weights
) {

  using Int = int32_t;
  Tensor output_tensor;

  auto empty_score_event_tensor = TPack<int64_t, 1, tmol::Device::CPU>::zeros({1});
  auto empty_annealer_event_tensor = TPack<int64_t, 1, tmol::Device::CPU>::zeros({1});

  TMOL_DISPATCH_FLOATING_DEVICE(
    context_coords.type(), "score_op", ([&] {
	using Real = scalar_t;
	constexpr tmol::Device Dev = device_t;

	auto output_tp = TPack<Real, 1, Dev>::zeros({alternate_coords.size(0)});
	auto output_tv = output_tp.view;

	LJLKRPEDispatch<common::ForallDispatch, Dev, Real, Int>::f(
          TCAST(context_coords),
          TCAST(context_block_type),
          TCAST(alternate_coords),
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
	  empty_annealer_event_tensor.view
	);

	/*
	LKRPEDispatch<common::ForallDispatch, Dev, Real, Int>::f(
          TCAST(context_coords),
          TCAST(context_block_type),
          TCAST(alternate_coords),
          TCAST(alternate_ids),
          TCAST(context_system_ids),
          TCAST(system_min_bond_separation),
          TCAST(system_inter_block_bondsep),
          TCAST(system_neighbor_list),
          TCAST(block_type_n_heavy_atoms),
	  TCAST(block_type_heavy_atom_inds),
          TCAST(block_type_atom_types),
          TCAST(block_type_n_interblock_bonds),
          TCAST(block_type_atoms_forming_chemical_bonds),
          TCAST(block_type_path_distance),
          TCAST(lk_type_params),
          TCAST(global_params),
          TCAST(lj_lk_weights),
	  output_tv
	);
	*/
	output_tensor = output_tp.tensor;
      }));

  return output_tensor;
}

Tensor
register_lj_lk_rotamer_pair_energy_eval(
  Tensor context_coords,
  Tensor context_block_type,
  Tensor alternate_coords,
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
  Tensor annealer
) {

  Tensor dummy_return_value;
  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
    context_coords.type(), "score_op", ([&] {
	using Real = scalar_t;
	constexpr tmol::Device Dev = device_t;

	LJLKRPERegistratorDispatch<common::ForallDispatch, Dev, Real, Int>::f(
          TCAST(context_coords),
          TCAST(context_block_type),
          TCAST(alternate_coords),
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
	  TCAST(annealer)
	);


	/*LKRPERegistratorDispatch<common::ForallDispatch, Dev, Real, Int>::f(
          TCAST(context_coords),
          TCAST(context_block_type),
          TCAST(alternate_coords),
          TCAST(alternate_ids),
          TCAST(context_system_ids),
          TCAST(system_min_bond_separation),
          TCAST(system_inter_block_bondsep),
          TCAST(system_neighbor_list),
          TCAST(block_type_n_heavy_atoms),
	  TCAST(block_type_heavy_atom_inds),
          TCAST(block_type_atom_types),
          TCAST(block_type_n_interblock_bonds),
          TCAST(block_type_atoms_forming_chemical_bonds),
          TCAST(block_type_path_distance),
          TCAST(lk_type_params),
          TCAST(global_params),
          TCAST(lj_lk_weights),
	  TCAST(output),
	  TCAST(annealer)
	);
	*/
      }));
  return dummy_return_value;
}


// template <
//     template <
//         template <tmol::Device>
//         class Dispatch,
//         tmol::Device D,
//         typename Real,
//         typename Int>
//     class ScoreDispatch,
//     template <tmol::Device>
//     class DispatchMethod>
// Tensor lk_score_op(
//     Tensor I,
//     Tensor atom_type_I,
//     Tensor heavyatom_inds_I,
//     Tensor J,
//     Tensor atom_type_J,
//     Tensor heavyatom_inds_J,
//     Tensor bonded_path_lengths,
//     Tensor type_params,
//     Tensor global_params) {
//   using tmol::utility::connect_backward_pass;
//   using tmol::utility::StackedSavedGradsBackward;
//
//   at::Tensor score;
//   at::Tensor dScore_dI;
//   at::Tensor dScore_dJ;
//
//   using Int = int64_t;
//
//   TMOL_DISPATCH_FLOATING_DEVICE(
//       I.type(), "score_op", ([&] {
//         using Real = scalar_t;
//         constexpr tmol::Device Dev = device_t;
//
//         auto result = ScoreDispatch<DispatchMethod, Dev, Real, Int>::f(
//             TCAST(I),
//             TCAST(atom_type_I),
//             TCAST(heavyatom_inds_I),
//             TCAST(J),
//             TCAST(atom_type_J),
//             TCAST(heavyatom_inds_J),
//             TCAST(bonded_path_lengths),
//             TCAST(type_params),
//             TCAST(global_params));
//
//         score = std::get<0>(result).tensor;
//         dScore_dI = std::get<1>(result).tensor;
//         dScore_dJ = std::get<2>(result).tensor;
//       }));
//
//   return connect_backward_pass({I, J}, score, [&]() {
//     return StackedSavedGradsBackward::create({dScore_dI, dScore_dJ});
//   });
// };


// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("score_ljlk_lj", &lj_score_op<LJDispatch, AABBDispatch>);
  m.def("score_ljlk_lj_triu", &lj_score_op<LJDispatch, AABBTriuDispatch>);
  m.def(
      "score_ljlk_lk_isotropic", &lk_score_op<LKIsotropicDispatch, AABBDispatch>);
  m.def(
      "score_ljlk_lk_isotropic_triu",
      &lk_score_op<LKIsotropicDispatch, AABBTriuDispatch>);
  m.def(
      "ljlk_pose_score",
      &ljlk_pose_score_op<ForeachDispatch>);
  m.def("score_ljlk_inter_system_scores", &rotamer_pair_energies_op);
  m.def("register_lj_lk_rotamer_pair_energy_eval",
      &register_lj_lk_rotamer_pair_energy_eval);
  m.def("ljlk_pose_scores", &ljlk_pose_scores);
}


}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
