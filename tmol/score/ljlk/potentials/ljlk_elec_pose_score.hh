#pragma once

#include <Eigen/Core>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

#include <tmol/score/elec/potentials/params.hh>
#include <tmol/score/ljlk/potentials/params.hh>

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using LJLKExternalVec = Eigen::Matrix<Real, N, 1>;

/// Whole-pose LJ/LK + electrostatics evaluation over one precomputed compact
/// block-neighbor list. The four unweighted score lanes remain separate so
/// score-term weights and decomposed-score APIs retain their existing meaning.
template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
struct LJLKAndElecPoseScoreDispatch {
  static auto forward(
      ContextManager& mgr,
      TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> pose_ind_for_atom,
      TView<Int, 2, D> first_rot_for_block,
      TView<Int, 2, D> first_rot_block_type,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      Int max_n_rots_per_pose,
      TView<Int, 3, D> pose_stack_min_bond_separation,
      TView<Int, 5, D> pose_stack_inter_block_bondsep,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Int, 2, D> block_type_atom_types,
      TView<Int, 1, D> block_type_n_interblock_bonds,
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
      TView<Int, 3, D> block_type_ljlk_path_distance,
      TView<Int, 1, D> block_type_is_ligand_fragment,
      TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
      TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
      TView<Real, 2, D> block_type_partial_charge,
      TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
      TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
      TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
          elec_global_params,
      TView<Int, 1, D> shared_compact_block_neighbors,
      bool require_gradient)
      -> std::tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>>;

  /// Evaluate the same four potentials while reducing them with live score
  /// weights inside the native traversal. The single output lane preserves
  /// ordinary autograd semantics and avoids four-lane derivative scratch.
  static auto forward_weighted(
      ContextManager& mgr,
      TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> pose_ind_for_atom,
      TView<Int, 2, D> first_rot_for_block,
      TView<Int, 2, D> first_rot_block_type,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      Int max_n_rots_per_pose,
      TView<Int, 3, D> pose_stack_min_bond_separation,
      TView<Int, 5, D> pose_stack_inter_block_bondsep,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Int, 2, D> block_type_atom_types,
      TView<Int, 1, D> block_type_n_interblock_bonds,
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
      TView<Int, 3, D> block_type_ljlk_path_distance,
      TView<Int, 1, D> block_type_is_ligand_fragment,
      TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
      TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
      TView<Real, 2, D> block_type_partial_charge,
      TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
      TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
      TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
          elec_global_params,
      TView<Int, 1, D> shared_compact_block_neighbors,
      TView<Real, 1, D> score_weights,
      bool require_gradient)
      -> std::tuple<TPack<Real, 4, D>, TPack<LJLKExternalVec<Real, 3>, 2, D>>;

  /// Evaluate weighted LJ/LK + electrostatics for a prepared sparse rotamer
  /// pair dispatch. Packing does not differentiate the energy table, so this
  /// compact path emits one live-weighted value per dispatch entry without
  /// allocating canonical score lanes or derivative scratch.
  static auto forward_weighted_rotamers(
      ContextManager& mgr,
      TView<LJLKExternalVec<Real, 3>, 1, D> rot_coords,
      TView<Int, 1, D> rot_coord_offset,
      TView<Int, 1, D> pose_ind_for_atom,
      TView<Int, 2, D> first_rot_for_block,
      TView<Int, 2, D> first_rot_block_type,
      TView<Int, 1, D> block_ind_for_rot,
      TView<Int, 1, D> pose_ind_for_rot,
      TView<Int, 1, D> block_type_ind_for_rot,
      TView<Int, 1, D> n_rots_for_pose,
      TView<Int, 1, D> rot_offset_for_pose,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      Int max_n_rots_per_pose,
      TView<Int, 3, D> pose_stack_min_bond_separation,
      TView<Int, 5, D> pose_stack_inter_block_bondsep,
      TView<Int, 1, D> block_type_n_atoms,
      TView<Int, 2, D> block_type_atom_types,
      TView<Int, 1, D> block_type_n_interblock_bonds,
      TView<Int, 2, D> block_type_atoms_forming_chemical_bonds,
      TView<Int, 3, D> block_type_ljlk_path_distance,
      TView<Int, 1, D> block_type_is_ligand_fragment,
      TView<LJLKTypeParams<Real>, 1, D> ljlk_type_params,
      TView<LJGlobalParams<Real>, 1, D> ljlk_global_params,
      TView<Real, 2, D> block_type_partial_charge,
      TView<Int, 3, D> block_type_elec_inter_repr_path_distance,
      TView<Int, 3, D> block_type_elec_intra_repr_path_distance,
      TView<tmol::score::elec::potentials::ElecGlobalParams<Real>, 1, D>
          elec_global_params,
      Real max_dis,
      TView<Real, 1, D> score_weights)
      -> std::tuple<
          TPack<Real, 4, D>,
          TPack<LJLKExternalVec<Real, 3>, 2, D>,
          TPack<Int, 2, D>>;

  /// Reduce score lanes with live weights in one device pass. The lane at
  /// fused_weight_begin is already weighted and represents fused_weight_width
  /// canonical lanes; all later lanes map past that canonical range.
  static auto reduce_weighted_scores(
      ContextManager& mgr,
      TView<Real, 2, D> score_lanes,
      TView<Real, 2, D> score_weights,
      Int fused_weight_begin,
      Int fused_weight_width) -> TPack<Real, 1, D>;

  /// Backward mapping for reduce_weighted_scores.
  static auto reduce_weighted_score_gradients(
      ContextManager& mgr,
      TView<Real, 1, D> output_gradient,
      TView<Real, 2, D> score_weights,
      Int n_score_lanes,
      Int fused_weight_begin,
      Int fused_weight_width) -> TPack<Real, 2, D>;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol
