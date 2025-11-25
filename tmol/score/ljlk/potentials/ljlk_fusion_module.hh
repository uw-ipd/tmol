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
#include <tmol/score/common/pose_score_fusion_module.hh>

#include "lj.hh"
#include "params.hh"
// #include "rotamer_pair_energy_lj.hh"

namespace tmol {
namespace score {
namespace ljlk {
namespace potentials {

template <
    template <tmol::Device> class DeviceOperations,
    tmol::Device D,
    typename Real,
    typename Int>
class LJLKPoseScoreFusionModule : public common::PoseScoreFusionModule {
 private:
  torch::Tensor rot_coord_offset_t_;
  torch::Tensor pose_ind_for_atom_t_;
  torch::Tensor first_rot_for_block_t_;
  torch::Tensor first_rot_block_type_t_;
  torch::Tensor block_ind_for_rot_t_;
  torch::Tensor pose_ind_for_rot_t_;
  torch::Tensor block_type_ind_for_rot_t_;
  torch::Tensor n_rots_for_pose_t_;
  torch::Tensor rot_offset_for_pose_t_;
  torch::Tensor n_rots_for_block_t_;
  torch::Tensor rot_offset_for_block_t_;
  Int max_n_rots_per_pose_;
  torch::Tensor pose_stack_min_bond_separation_t_;
  torch::Tensor pose_stack_inter_block_bondsep_t_;
  torch::Tensor block_type_n_atoms_t_;
  torch::Tensor block_type_n_heavy_atoms_in_tile_t_;
  torch::Tensor block_type_heavy_atoms_in_tile_t_;
  torch::Tensor block_type_atom_types_t_;
  torch::Tensor block_type_n_interblock_bonds_t_;
  torch::Tensor block_type_atoms_forming_chemical_bonds_t_;
  torch::Tensor block_type_path_distance_t_;

  // LJ parameters
  torch::Tensor type_params_t_;
  torch::Tensor global_params_t_;
  bool output_block_pair_energies_;

  // do we need to compute gradients?
  bool require_gradient_;

  torch::Tensor scratch_rot_spheres_t_;
  torch::Tensor scratch_rot_neighbors_t_;
  torch::Tensor scratch_rot_neighbors_offset_t_;
  torch::Tensor dV_dcoords_t_;
  torch::Tensor dispatch_indices_t_;
  std::vector<void*>
      events_;            // for cuda events; we need pointers to cudaEvent_t
                          // for forward and backward passes that we'll wait on
  void* n_dispatch_ptr_;  // IDEA: we need a mgpu::mem_t for CUDA and just a
                          // wimpy int on CPU
 public:
  LJLKPoseScoreFusionModule(
      torch::Tensor rot_coord_offset,
      torch::Tensor pose_ind_for_atom,
      torch::Tensor first_rot_for_block,
      torch::Tensor first_rot_block_type,
      torch::Tensor block_ind_for_rot,
      torch::Tensor pose_ind_for_rot,
      torch::Tensor block_type_ind_for_rot,
      torch::Tensor n_rots_for_pose,
      torch::Tensor rot_offset_for_pose,
      torch::Tensor n_rots_for_block,
      torch::Tensor rot_offset_for_block,
      Int max_n_rots_per_pose,
      torch::Tensor pose_stack_min_bond_separation,
      torch::Tensor pose_stack_inter_block_bondsep,
      torch::Tensor block_type_n_atoms,
      torch::Tensor block_type_n_heavy_atoms_in_tile,
      torch::Tensor block_type_heavy_atoms_in_tile,
      torch::Tensor block_type_atom_types,
      torch::Tensor block_type_n_interblock_bonds,
      torch::Tensor block_type_atoms_forming_chemical_bonds,
      torch::Tensor block_type_path_distance,
      torch::Tensor type_params,
      torch::Tensor global_params,
      bool output_block_pair_energies,
      bool require_gradient);

  ~LJLKPoseScoreFusionModule() override;

  void prepare_for_scoring(torch::Tensor coords) override;
  void forward(torch::Tensor coords, torch::Tensor V) override;
  void backward(
      torch::Tensor coords, torch::Tensor dTdV, torch::Tensor dVdxyz) override;

  int n_terms() const override;
  int n_poses() const override;
  int max_n_blocks() const override;
  bool output_block_pair_energies() const override;

  int n_atoms() const;
  int n_rotamers() const;
};

}  // namespace potentials
}  // namespace ljlk
}  // namespace score
}  // namespace tmol