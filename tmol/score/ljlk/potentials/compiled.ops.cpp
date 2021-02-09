#include <torch/script.h>
#include <tmol/utility/autograd.hh>

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
  using tmol::utility::connect_backward_pass;
  using tmol::utility::StackedSavedGradsBackward;

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

  return connect_backward_pass({I, J}, score, [&]() {
    return StackedSavedGradsBackward::create({dScore_dI, dScore_dJ});
  });
};

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
  Tensor block_type_atom_types,
  Tensor block_type_heavy_atom_inds,
  Tensor block_type_n_interblock_bonds,
  Tensor block_type_atoms_forming_chemical_bonds,
  Tensor block_type_path_distance,
  Tensor lj_type_params,
  Tensor lk_type_params,
  Tensor global_params,
  Tensor lj_lk_weights
) {
  
  using Int = int32_t;
  Tensor output_tensor;

  TMOL_DISPATCH_FLOATING_DEVICE(
    context_coords.type(), "score_op", ([&] {
	using Real = scalar_t;
	constexpr tmol::Device Dev = device_t;

	auto output_tp = TPack<Real, 1, Dev>::zeros({alternate_coords.size(0)});
	auto output_tv = output_tp.view;

	LJRPEDispatch<common::ForallDispatch, Dev, Real, Int>::f(
          TCAST(context_coords),
          TCAST(context_block_type),
          TCAST(alternate_coords),
          TCAST(alternate_ids),
          TCAST(context_system_ids),
          TCAST(system_min_bond_separation),
          TCAST(system_inter_block_bondsep),
          TCAST(system_neighbor_list),
          TCAST(block_type_n_atoms),
          TCAST(block_type_atom_types),
          TCAST(block_type_n_interblock_bonds),
          TCAST(block_type_atoms_forming_chemical_bonds),
          TCAST(block_type_path_distance),
          TCAST(lj_type_params),
          TCAST(global_params),
	  TCAST(lj_lk_weights),
	  output_tv
	);

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
  Tensor block_type_atom_types,
  Tensor block_type_heavy_atom_inds,
  Tensor block_type_n_interblock_bonds,
  Tensor block_type_atoms_forming_chemical_bonds,
  Tensor block_type_path_distance,
  Tensor lj_type_params,
  Tensor lk_type_params,
  Tensor global_params,
  Tensor lj_lk_weights,
  Tensor output,
  Tensor annealer
) {

  Tensor dummy_return_value;
  using Int = int32_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
    context_coords.type(), "score_op", ([&] {
	using Real = scalar_t;
	constexpr tmol::Device Dev = device_t;
	
	LJRPERegistratorDispatch<common::ForallDispatch, Dev, Real, Int>::f(
          TCAST(context_coords),
          TCAST(context_block_type),
          TCAST(alternate_coords),
          TCAST(alternate_ids),
          TCAST(context_system_ids),
          TCAST(system_min_bond_separation),
          TCAST(system_inter_block_bondsep),
          TCAST(system_neighbor_list),
          TCAST(block_type_n_atoms),
          TCAST(block_type_atom_types),
          TCAST(block_type_n_interblock_bonds),
          TCAST(block_type_atoms_forming_chemical_bonds),
          TCAST(block_type_path_distance),
          TCAST(lj_type_params),
          TCAST(global_params),
	  TCAST(lj_lk_weights),
	  TCAST(output),
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
  using tmol::utility::connect_backward_pass;
  using tmol::utility::StackedSavedGradsBackward;

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

  return connect_backward_pass({I, J}, score, [&]() {
    return StackedSavedGradsBackward::create({dScore_dI, dScore_dJ});
  });
};


static auto registry =
    torch::jit::RegisterOperators()
        .op("tmol::score_ljlk_lj", &lj_score_op<LJDispatch, AABBDispatch>)
        .op("tmol::score_ljlk_lj_triu", &lj_score_op<LJDispatch, AABBTriuDispatch>)
        .op("tmol::score_ljlk_lk_isotropic",
            &lk_score_op<LKIsotropicDispatch, AABBDispatch>)
        .op("tmol::score_ljlk_lk_isotropic_triu",
            &lk_score_op<LKIsotropicDispatch, AABBTriuDispatch>)
        .op("tmol::score_ljlk_inter_system_scores", &rotamer_pair_energies_op)
        .op("tmol::register_lj_lk_rotamer_pair_energy_eval",
            &register_lj_lk_rotamer_pair_energy_eval);


}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
