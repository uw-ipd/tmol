#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/sphere_overlap.impl.hh>
#include <tmol/score/common/tile_atom_pair_evaluation.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/upper_triangle_indices.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/ljlk/potentials/ljlk_fusion_module.hh>

#include <tmol/score/ljlk/potentials/lj.hh>
#include <tmol/score/ljlk/potentials/ljlk.hh>
#include <tmol/score/ljlk/potentials/ljlk_scoring_macros.hh>
#include <tmol/score/ljlk/potentials/ljlk_pose_score.hh>
#include <tmol/score/ljlk/potentials/lk_isotropic.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

// The maximum number of inter-residue chemical bonds
#define MAX_N_CONN 4
#define TILE_SIZE 32

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

using torch::Tensor;

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::
    LJLKPoseScoreFusionModule(
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
        Int max_n_rots_per_pose,
        Tensor pose_stack_min_bond_separation,
        Tensor pose_stack_inter_block_bondsep,
        Tensor block_type_n_atoms,
        Tensor block_type_n_heavy_atoms_in_tile,
        Tensor block_type_heavy_atoms_in_tile,
        Tensor block_type_atom_types,
        Tensor block_type_n_interblock_bonds,
        Tensor block_type_atoms_forming_chemical_bonds,
        Tensor block_type_path_distance,
        // LJ parameter,
        Tensor type_params,
        Tensor global_params,
        bool output_block_pair_energies,
        // do we need to compute gradients,
        bool require_gradient)
    : rot_coord_offset_t_(rot_coord_offset),
      pose_ind_for_atom_t_(pose_ind_for_atom),
      first_rot_for_block_t_(first_rot_for_block),
      first_rot_block_type_t_(first_rot_block_type),
      block_ind_for_rot_t_(block_ind_for_rot),
      pose_ind_for_rot_t_(pose_ind_for_rot),
      block_type_ind_for_rot_t_(block_type_ind_for_rot),
      n_rots_for_pose_t_(n_rots_for_pose),
      rot_offset_for_pose_t_(rot_offset_for_pose),
      n_rots_for_block_t_(n_rots_for_block),
      rot_offset_for_block_t_(rot_offset_for_block),
      max_n_rots_per_pose_(max_n_rots_per_pose),
      pose_stack_min_bond_separation_t_(pose_stack_min_bond_separation),
      pose_stack_inter_block_bondsep_t_(pose_stack_inter_block_bondsep),
      block_type_n_atoms_t_(block_type_n_atoms),
      block_type_n_heavy_atoms_in_tile_t_(block_type_n_heavy_atoms_in_tile),
      block_type_heavy_atoms_in_tile_t_(block_type_heavy_atoms_in_tile),
      block_type_atom_types_t_(block_type_atom_types),
      block_type_n_interblock_bonds_t_(block_type_n_interblock_bonds),
      block_type_atoms_forming_chemical_bonds_t_(
          block_type_atoms_forming_chemical_bonds),
      block_type_path_distance_t_(block_type_path_distance),
      type_params_t_(type_params),
      global_params_t_(global_params),
      output_block_pair_energies_(output_block_pair_energies),
      require_gradient_(require_gradient) {
  int const n_poses = first_rot_for_block_t_.size(0);
  int const n_atoms = pose_ind_for_atom_t_.size(0);
  int const max_n_blocks = first_rot_for_block_t_.size(1);
  int const n_rotamers = block_ind_for_rot_t_.size(0);

  scratch_rot_spheres_t_ = first_rot_for_block_t_.new_zeros(
      {n_poses, max_n_blocks, 4}, torch::kFloat);
  scratch_rot_neighbors_t_ = first_rot_for_block_t_.new_zeros(
      {n_poses, max_n_blocks, max_n_blocks}, torch::kInt32);
  scratch_rot_neighbors_offset_t_ = first_rot_for_block_t_.new_zeros(
      {n_poses * max_n_blocks * max_n_blocks}, torch::kInt32);
  dV_dcoords_t_ =
      first_rot_for_block_t_.new_zeros({3, n_atoms, 3}, torch::kFloat);
  dispatch_indices_t_ = first_rot_for_block_t_.new_zeros(
      {3, n_poses * max_n_blocks * max_n_blocks}, torch::kInt32);

  events_.resize(1);

  n_dispatch_ptr_ =
      DeviceOperations<D>::template allocate_scan_total_storage<Int>();
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::
    ~LJLKPoseScoreFusionModule() {
  printf("LJLKPoseScoreFusionModule destructor\n");
  if (events_[0]) {
    DeviceOperations<D>::deallocate_synchronization_event(events_[0]);
    events_[0] = 0;
  }
  if (n_dispatch_ptr_) {
    DeviceOperations<D>::template deallocate_scan_total_storage<Int>(
        n_dispatch_ptr_);
    n_dispatch_ptr_ = 0;
  }
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
void LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::
    prepare_for_scoring(Tensor rot_coords_t) {
  printf("LJLKPoseScoreFusionModule::prepare_for_scoring\n");

  // Zero initialize the arrays we need in order to figure out which
  // pairs of residues are interacting
  DeviceOperations<D>::template set_zero<Int>(
      dispatch_indices_t_.data_ptr<Int>(), dispatch_indices_t_.numel());
  DeviceOperations<D>::template set_zero<Int>(
      scratch_rot_neighbors_t_.data_ptr<Int>(),
      scratch_rot_neighbors_t_.numel());
  DeviceOperations<D>::template set_zero<Int>(
      scratch_rot_neighbors_offset_t_.data_ptr<Int>(),
      scratch_rot_neighbors_offset_t_.numel());

  score::common::sphere_overlap::
      compute_rot_spheres<DeviceOperations, D, Real, Int>::f(
          TCAST(rot_coords_t),
          TCAST(rot_coord_offset_t_),
          TCAST(block_type_ind_for_rot_t_),
          TCAST(block_type_n_atoms_t_),
          TCAST(scratch_rot_spheres_t_));

  score::common::sphere_overlap::
      detect_rot_neighbors<DeviceOperations, D, Real, Int>::f(
          max_n_rots_per_pose_,
          TCAST(block_ind_for_rot_t_),
          TCAST(block_type_ind_for_rot_t_),
          TCAST(block_type_n_atoms_t_),
          TCAST(n_rots_for_pose_t_),
          TCAST(rot_offset_for_pose_t_),
          TCAST(n_rots_for_block_t_),
          TCAST(scratch_rot_spheres_t_),
          TCAST(scratch_rot_neighbors_t_),
          Real(5.5));  // 5.5A hard coded here. Please fix! TEMP!

  events_[0] = DeviceOperations<D>::allocate_synchronization_event();

  // TO DO: Replace this call with one that submits the
  // scan task and returns a cudaEvent that we'll hold on to
  score::common::sphere_overlap::
      asynch_block_neighbor_indices<DeviceOperations, D, Int>::f(
          TCAST(scratch_rot_neighbors_t_),
          TCAST(rot_offset_for_pose_t_),
          TCAST(dispatch_indices_t_),
          events_[0],
          n_dispatch_ptr_);

  // Zero initialize the remaining things
  DeviceOperations<D>::template set_zero<Real>(
      dV_dcoords_t_.data_ptr<Real>(), dV_dcoords_t_.numel());
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
void LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::forward(
    Tensor coords_t, Tensor V_t) {
  printf("LJLKPoseScoreFusionModule::forward\n");
  printf(
      "V size: %d %d %d %d\n",
      V_t.size(0),
      V_t.size(1),
      V_t.size(2),
      V_t.size(3));

  // Give these variables names that the macros from ljlk_scoring_macros.hh
  // expect.
  auto rot_coords = view_tensor<Vec<Real, 3>, 1, D>(coords_t);
  auto rot_coord_offset = view_tensor<Int, 1, D>(rot_coord_offset_t_);
  auto pose_ind_for_atom = view_tensor<Int, 1, D>(pose_ind_for_atom_t_);
  auto first_rot_for_block = view_tensor<Int, 2, D>(first_rot_for_block_t_);
  auto first_rot_block_type = view_tensor<Int, 2, D>(first_rot_block_type_t_);
  auto block_ind_for_rot = view_tensor<Int, 1, D>(block_ind_for_rot_t_);
  auto pose_ind_for_rot = view_tensor<Int, 1, D>(pose_ind_for_rot_t_);
  auto block_type_ind_for_rot =
      view_tensor<Int, 1, D>(block_type_ind_for_rot_t_);
  auto n_rots_for_pose = view_tensor<Int, 1, D>(n_rots_for_pose_t_);
  auto rot_offset_for_pose = view_tensor<Int, 1, D>(rot_offset_for_pose_t_);
  auto n_rots_for_block = view_tensor<Int, 2, D>(n_rots_for_block_t_);
  auto rot_offset_for_block = view_tensor<Int, 2, D>(rot_offset_for_block_t_);
  auto pose_stack_min_bond_separation =
      view_tensor<Int, 3, D>(pose_stack_min_bond_separation_t_);
  auto pose_stack_inter_block_bondsep =
      view_tensor<Int, 5, D>(pose_stack_inter_block_bondsep_t_);
  auto block_type_n_atoms = view_tensor<Int, 1, D>(block_type_n_atoms_t_);
  auto block_type_n_heavy_atoms_in_tile =
      view_tensor<Int, 2, D>(block_type_n_heavy_atoms_in_tile_t_);
  auto block_type_heavy_atoms_in_tile =
      view_tensor<Int, 2, D>(block_type_heavy_atoms_in_tile_t_);
  auto block_type_atom_types = view_tensor<Int, 2, D>(block_type_atom_types_t_);
  auto block_type_n_interblock_bonds =
      view_tensor<Int, 1, D>(block_type_n_interblock_bonds_t_);
  auto block_type_atoms_forming_chemical_bonds =
      view_tensor<Int, 2, D>(block_type_atoms_forming_chemical_bonds_t_);
  auto block_type_path_distance =
      view_tensor<Int, 3, D>(block_type_path_distance_t_);
  auto type_params = view_tensor<LJLKTypeParams<Real>, 1, D>(type_params_t_);
  auto global_params =
      view_tensor<LJGlobalParams<Real>, 1, D>(global_params_t_);
  auto dV_dcoords = view_tensor<Vec<Real, 3>, 2, D>(dV_dcoords_t_);
  auto dispatch_indices = view_tensor<Int, 2, D>(dispatch_indices_t_);

  // rename for the sake of capture
  bool const output_block_pair_energies = output_block_pair_energies_;
  bool const require_gradient = require_gradient_;

  auto output = view_tensor<Real, 4, D>(V_t);

  /////////////////////////////

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_energies_sparse_dispatch = ([=] TMOL_DEVICE_FUNC(int cta) {
    auto atom_pair_lj_fn = ([=] ATOM_PAIR_LJ_SCORE_W_GRADIENT_EVAL);

    auto atom_pair_lk_fn = ([=] ATOM_PAIR_LK_SCORE_W_GRADIENT_EVAL);

    auto score_inter_lj_atom_pair =
        ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_intra_lj_atom_pair =
        ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_lj_fn));

    auto score_inter_lk_atom_pair =
        ([=] SCORE_INTER_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto score_intra_lk_atom_pair =
        ([=] SCORE_INTRA_LK_ATOM_PAIR(atom_pair_lk_fn));

    auto load_block_coords_and_params_into_shared =
        ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

    auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    Real total_ljatr = 0;
    Real total_ljrep = 0;
    Real total_lk = 0;

    int const max_important_bond_separation = 4;

    int const pose_ind = dispatch_indices[0][cta];

    int const rot_ind1 = dispatch_indices[1][cta];
    int const rot_ind2 = dispatch_indices[2][cta];

    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_ind2 = block_ind_for_rot[rot_ind2];

    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const block_type2 = block_type_ind_for_rot[rot_ind2];

    int const n_atoms1 = block_type_n_atoms[block_type1];
    int const n_atoms2 = block_type_n_atoms[block_type2];

    if (block_type1 < 0 || block_type2 < 0) {
      return;
    }

    auto load_tile_invariant_interres_data =
        ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

    auto load_interres1_tile_data_to_shared =
        ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

    auto load_interres2_tile_data_to_shared =
        ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

    auto load_interres_data_from_shared = ([=] LOAD_INTERRES_DATA_FROM_SHARED);

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_interres_atom_pair_scores = ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

    auto store_calculated_energies = ([=] STORE_POSE_CALCULATED_ENERGIES);

    auto load_tile_invariant_intrares_data =
        ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

    auto load_intrares1_tile_data_to_shared =
        ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

    auto load_intrares2_tile_data_to_shared =
        ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

    auto load_intrares_data_from_shared = ([=] LOAD_INTRARES_DATA_FROM_SHARED);

    // Evaluate both the LJ and LK scores in separate dispatches
    // over all atoms in the tile and the subset of heavy atoms in
    // the tile
    auto eval_intrares_atom_pair_scores = ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

    tmol::score::common::tile_evaluate_rot_pair<
        DeviceOperations,
        D,
        LJLKScoringData<Real>,
        LJLKScoringData<Real>,
        Real,
        TILE_SIZE>(
        shared,
        pose_ind,
        rot_ind1,
        rot_ind2,
        block_ind1,
        block_ind2,
        block_type1,
        block_type2,
        n_atoms1,
        n_atoms2,
        load_tile_invariant_interres_data,
        load_interres1_tile_data_to_shared,
        load_interres2_tile_data_to_shared,
        load_interres_data_from_shared,
        eval_interres_atom_pair_scores,
        store_calculated_energies,
        load_tile_invariant_intrares_data,
        load_intrares1_tile_data_to_shared,
        load_intrares2_tile_data_to_shared,
        load_intrares_data_from_shared,
        eval_intrares_atom_pair_scores,
        store_calculated_energies);
  });

  // Wait on the event that signals that the number of interacting block pairs
  // is ready to be read from
  DeviceOperations<D>::synchronize_on_event(events_[0]);

  // Now that we're done with this event, we need to deallocate it
  DeviceOperations<D>::deallocate_synchronization_event(events_[0]);
  events_[0] = nullptr;  // reset the event handle

  int const n_dispatch =
      DeviceOperations<D>::template read_scan_total<Int>(n_dispatch_ptr_);

  DeviceOperations<D>::template foreach_workgroup<launch_t>(
      n_dispatch, eval_energies_sparse_dispatch);
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
void LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::backward(
    Tensor coords_t, Tensor dTdV_t, Tensor dVdxyz_t) {
  printf("LJLKPoseScoreFusionModule::backward\n");
  printf("dTdV_t size: %d %d\n", dTdV_t.size(0), dTdV_t.size(1));
  printf("dVdxyz_t size: %d %d\n", dVdxyz_t.size(0), dVdxyz_t.size(1));
  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  if (!output_block_pair_energies_) {
    auto pose_ind_for_atom = view_tensor<Int, 1, D>(pose_ind_for_atom_t_);
    auto dTdV = view_tensor<Real, 2, D>(dTdV_t);
    // The output gradient tensor
    auto dVdxyz = view_tensor<Vec<Real, 3>, 1, D>(dVdxyz_t);
    // Computed in the forward pass
    auto dV_dcoords = view_tensor<Vec<Real, 3>, 2, D>(dV_dcoords_t_);
    // Pose scoring: on the forward pass, we calculated dVdxyz and now
    // we need to weight them by dTdV to get dT/dxyz
    auto increment_derivs = ([=] TMOL_DEVICE_FUNC(int atom_id) {
      Int pose_ind = pose_ind_for_atom[atom_id];
      for (int i = 0; i < 3; i++) {
        Real coord_i_grad_contribution = 0;
        for (int j = 0; j < 3; j++) {
          coord_i_grad_contribution +=
              dV_dcoords[j][atom_id][i] * dTdV[j][pose_ind];
        }
        accumulate<D, Real>::add(dVdxyz[atom_id][i], coord_i_grad_contribution);
      }
    });
    DeviceOperations<D>::template forall<launch_t>(n_atoms(), increment_derivs);
  } else {
    // block pair derivative evaluation

    auto rot_coords = view_tensor<Vec<Real, 3>, 1, D>(coords_t);
    auto rot_coord_offset = view_tensor<Int, 1, D>(rot_coord_offset_t_);
    auto pose_ind_for_atom = view_tensor<Int, 1, D>(pose_ind_for_atom_t_);
    auto first_rot_for_block = view_tensor<Int, 2, D>(first_rot_for_block_t_);
    auto first_rot_block_type = view_tensor<Int, 2, D>(first_rot_block_type_t_);
    auto block_ind_for_rot = view_tensor<Int, 1, D>(block_ind_for_rot_t_);
    auto pose_ind_for_rot = view_tensor<Int, 1, D>(pose_ind_for_rot_t_);
    auto block_type_ind_for_rot =
        view_tensor<Int, 1, D>(block_type_ind_for_rot_t_);
    auto n_rots_for_pose = view_tensor<Int, 1, D>(n_rots_for_pose_t_);
    auto rot_offset_for_pose = view_tensor<Int, 1, D>(rot_offset_for_pose_t_);
    auto n_rots_for_block = view_tensor<Int, 2, D>(n_rots_for_block_t_);
    auto rot_offset_for_block = view_tensor<Int, 2, D>(rot_offset_for_block_t_);
    auto pose_stack_min_bond_separation =
        view_tensor<Int, 3, D>(pose_stack_min_bond_separation_t_);
    auto pose_stack_inter_block_bondsep =
        view_tensor<Int, 5, D>(pose_stack_inter_block_bondsep_t_);
    auto block_type_n_atoms = view_tensor<Int, 1, D>(block_type_n_atoms_t_);
    auto block_type_n_heavy_atoms_in_tile =
        view_tensor<Int, 2, D>(block_type_n_heavy_atoms_in_tile_t_);
    auto block_type_heavy_atoms_in_tile =
        view_tensor<Int, 2, D>(block_type_heavy_atoms_in_tile_t_);
    auto block_type_atom_types =
        view_tensor<Int, 2, D>(block_type_atom_types_t_);
    auto block_type_n_interblock_bonds =
        view_tensor<Int, 1, D>(block_type_n_interblock_bonds_t_);
    auto block_type_atoms_forming_chemical_bonds =
        view_tensor<Int, 2, D>(block_type_atoms_forming_chemical_bonds_t_);
    auto block_type_path_distance =
        view_tensor<Int, 3, D>(block_type_path_distance_t_);
    auto type_params = view_tensor<LJLKTypeParams<Real>, 1, D>(type_params_t_);
    auto global_params =
        view_tensor<LJGlobalParams<Real>, 1, D>(global_params_t_);
    // auto dV_dcoords = view_tensor<Vec<Real, 3>, 2, D>(dV_dcoords_t_);
    auto dispatch_indices = view_tensor<Int, 2, D>(dispatch_indices_t_);
    auto dTdV = view_tensor<Real, 2, D>(dTdV_t);
    auto dV_dcoords = view_tensor<Vec<Real, 3>, 1, D>(dVdxyz_t);

    // rename for the sake of capture
    bool const output_block_pair_energies = output_block_pair_energies_;

    auto eval_derivs = ([=] TMOL_DEVICE_FUNC(int cta) {
      auto atom_pair_lj_fn = ([=] ATOM_PAIR_LJ_SPARSE_IND_SCORING_DERIVS_EVAL);

      auto atom_pair_lk_fn = ([=] ATOM_PAIR_LK_SPARSE_IND_SCORING_DERIVS_EVAL);

      auto score_inter_lj_atom_pair =
          ([=] SCORE_INTER_LJ_ATOM_PAIR(atom_pair_lj_fn));

      auto score_intra_lj_atom_pair =
          ([=] SCORE_INTRA_LJ_ATOM_PAIR(atom_pair_lj_fn));

      auto score_inter_lk_atom_pair =
          ([=] SCORE_INTER_LK_ATOM_PAIR(atom_pair_lk_fn));

      auto score_intra_lk_atom_pair =
          ([=] SCORE_INTRA_LK_ATOM_PAIR(atom_pair_lk_fn));

      auto load_block_coords_and_params_into_shared =
          ([=] LOAD_BLOCK_COORDS_AND_PARAMS_INTO_SHARED);

      auto load_block_into_shared = ([=] LOAD_BLOCK_INTO_SHARED);

      SHARED_MEMORY union shared_mem_union {
        shared_mem_union() {}
        LJLKBlockPairSharedData<Real, TILE_SIZE, MAX_N_CONN> m;
        CTA_REAL_REDUCE_T_VARIABLE;

      } shared;

      Real total_lj = 0;
      Real total_lk = 0;

      int const max_important_bond_separation = 4;

      int const pose_ind = dispatch_indices[0][cta];

      int const rot_ind1 = dispatch_indices[1][cta];
      int const rot_ind2 = dispatch_indices[2][cta];

      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const block_ind2 = block_ind_for_rot[rot_ind2];

      int const block_type1 = block_type_ind_for_rot[rot_ind1];
      int const block_type2 = block_type_ind_for_rot[rot_ind2];

      if (block_type1 < 0 || block_type2 < 0) {
        return;
      }

      int const n_atoms1 = block_type_n_atoms[block_type1];
      int const n_atoms2 = block_type_n_atoms[block_type2];

      auto load_tile_invariant_interres_data =
          ([=] LOAD_TILE_INVARIANT_INTERRES_DATA);

      auto load_interres1_tile_data_to_shared =
          ([=] LOAD_INTERRES1_TILE_DATA_TO_SHARED);

      auto load_interres2_tile_data_to_shared =
          ([=] LOAD_INTERRES2_TILE_DATA_TO_SHARED);

      auto load_interres_data_from_shared =
          ([=] LOAD_INTERRES_DATA_FROM_SHARED);

      auto eval_interres_atom_pair_scores =
          ([=] EVAL_INTERRES_ATOM_PAIR_SCORES);

      auto store_calculated_energies =
          ([=](LJLKScoringData<Real>& score_dat, shared_mem_union& shared) {
            ;  // no op for gradients ()
          });

      auto load_tile_invariant_intrares_data =
          ([=] LOAD_TILE_INVARIANT_INTRARES_DATA);

      auto load_intrares1_tile_data_to_shared =
          ([=] LOAD_INTRARES1_TILE_DATA_TO_SHARED);

      auto load_intrares2_tile_data_to_shared =
          ([=] LOAD_INTRARES2_TILE_DATA_TO_SHARED);

      auto load_intrares_data_from_shared =
          ([=] LOAD_INTRARES_DATA_FROM_SHARED);

      auto eval_intrares_atom_pair_scores =
          ([=] EVAL_INTRARES_ATOM_PAIR_SCORES);

      tmol::score::common::tile_evaluate_rot_pair<
          DeviceOperations,
          D,
          LJLKScoringData<Real>,
          LJLKScoringData<Real>,
          Real,
          TILE_SIZE>(
          shared,
          pose_ind,
          rot_ind1,
          rot_ind2,
          block_ind1,
          block_ind2,
          block_type1,
          block_type2,
          n_atoms1,
          n_atoms2,
          load_tile_invariant_interres_data,
          load_interres1_tile_data_to_shared,
          load_interres2_tile_data_to_shared,
          load_interres_data_from_shared,
          eval_interres_atom_pair_scores,
          store_calculated_energies,
          load_tile_invariant_intrares_data,
          load_intrares1_tile_data_to_shared,
          load_intrares2_tile_data_to_shared,
          load_intrares_data_from_shared,
          eval_intrares_atom_pair_scores,
          store_calculated_energies);
    });

    ///////////////////////////////////////////////////////////////////////

    // Three steps
    // 0: setup
    // 1: launch a kernel to find a small bounding sphere surrounding the blocks
    // 2: launch a kernel to look for spheres that are within striking distance
    // of each other 3: launch a kernel to evaluate lj/lk between pairs of
    // blocks within striking distance
    int const n_dispatch =
        DeviceOperations<D>::template read_scan_total<Int>(n_dispatch_ptr_);

    DeviceOperations<D>::template foreach_workgroup<launch_t>(
        n_dispatch, eval_derivs);
  }
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
int LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::n_terms() const {
  return 3;  // LJatr, LJrep, + LKiso
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
int LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::n_poses() const {
  return first_rot_for_block_t_.size(0);
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
int LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::max_n_blocks()
    const {
  return first_rot_for_block_t_.size(1);
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
bool LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::
    output_block_pair_energies() const {
  return output_block_pair_energies_;
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
int LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::n_atoms() const {
  return pose_ind_for_atom_t_.size(0);
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
int LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::n_rotamers()
    const {
  return block_ind_for_rot_t_.size(0);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol