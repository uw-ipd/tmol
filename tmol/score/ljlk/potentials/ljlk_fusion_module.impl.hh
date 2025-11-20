#pragma once
#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/tuple.hh>

#include "ljlk_fusion_module.hh"
#include "lj.hh"
#include "params.hh"
// #include "rotamer_pair_energy_lj.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

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
    : rot_coord_offset_(rot_coord_offset),
      pose_ind_for_atom_(pose_ind_for_atom),
      first_rot_for_block_(first_rot_for_block),
      first_rot_block_type_(first_rot_block_type),
      block_ind_for_rot_(block_ind_for_rot),
      pose_ind_for_rot_(pose_ind_for_rot),
      block_type_ind_for_rot_(block_type_ind_for_rot),
      n_rots_for_pose_(n_rots_for_pose),
      rot_offset_for_pose_(rot_offset_for_pose),
      n_rots_for_block_(n_rots_for_block),
      rot_offset_for_block_(rot_offset_for_block),
      max_n_rots_per_pose_(max_n_rots_per_pose),
      pose_stack_min_bond_separation_(pose_stack_min_bond_separation),
      pose_stack_inter_block_bondsep_(pose_stack_inter_block_bondsep),
      block_type_n_atoms_(block_type_n_atoms),
      block_type_n_heavy_atoms_in_tile_(block_type_n_heavy_atoms_in_tile),
      block_type_heavy_atoms_in_tile_(block_type_heavy_atoms_in_tile),
      block_type_atom_types_(block_type_atom_types),
      block_type_n_interblock_bonds_(block_type_n_interblock_bonds),
      block_type_atoms_forming_chemical_bonds_(
          block_type_atoms_forming_chemical_bonds),
      block_type_path_distance_(block_type_path_distance),
      type_params_(type_params),
      global_params_(global_params),
      output_block_pair_energies_(output_block_pair_energies),
      require_gradient_(require_gradient) {
  int const n_poses = first_rot_for_block_.size(0);
  int const n_atoms = pose_ind_for_atom_.size(0);
  int const max_n_blocks = first_rot_for_block_.size(1);
  int const n_rotamers = block_ind_for_rot_.size(0);

  scratch_rot_spheres_ =
      first_rot_for_block_.new_zeros({n_poses, max_n_blocks, 4}, torch::kFloat);
  scratch_rot_neighbors_ = first_rot_for_block_.new_zeros(
      {n_poses, max_n_blocks, max_n_blocks}, torch::kInt32);
  scratch_rot_neighbors_offset_ = first_rot_for_block_.new_zeros(
      {n_poses * max_n_blocks * max_n_blocks}, torch::kInt32);
  dispatch_indices_ = first_rot_for_block_.new_zeros(
      {n_poses * max_n_blocks * max_n_blocks}, torch::kInt32);
  events_ = first_rot_for_block_.new_zeros(
      {2}, torch::kInt64);    // cudaEvent_t is pointer-sized
  n_dispatch_ptr_ = nullptr;  // TEMP!
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::
    ~LJLKPoseScoreFusionModule() {
  printf("LJLKPoseScoreFusionModule destructor\n");
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
void LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::
    prepare_for_scoring(Tensor coords) {
  printf("LJLKPoseScoreFusionModule::prepare_for_scoring\n");
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
void LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::forward(
    Tensor coords, Tensor V) {
  printf("LJLKPoseScoreFusionModule::forward\n");
  printf("V size: %d %d %d %d\n", V.size(0), V.size(1), V.size(2), V.size(3));
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
void LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::backward(
    Tensor coords, Tensor dTdV, Tensor dVdxyz) {
  printf("LJLKPoseScoreFusionModule::backward\n");
  printf("dTdV size: %d %d\n", dTdV.size(0), dTdV.size(1));
  printf(
      "dVdxyz size: %d %d %d %d\n",
      dVdxyz.size(0),
      dVdxyz.size(1),
      dVdxyz.size(2));
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
  return first_rot_for_block_.size(0);
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
int LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::max_n_blocks()
    const {
  return first_rot_for_block_.size(1);
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
  return pose_ind_for_atom_.size(0);
}

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
int LJLKPoseScoreFusionModule<DeviceOperations, D, Real, Int>::n_rotamers()
    const {
  return block_ind_for_rot_.size(0);
}

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol