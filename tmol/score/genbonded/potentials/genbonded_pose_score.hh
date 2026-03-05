#pragma once

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

#include "params.hh"

namespace tmol {
namespace score {
namespace genbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

// ---------------------------------------------------------------------------
// Pose-level scoring dispatch
//
// Uses the same rotamer-style coordinate convention as
// CartBondedPoseScoreDispatch:
//   rot_coords       – flat [N_atoms, 3] coordinate tensor
//   rot_coord_offset – per-rotamer offset into rot_coords
//   first_rot_for_block / first_rot_block_type – pose × block → rotamer index
//
// Term-specific tensors (from GenBondedEnergyTerm.get_score_term_attributes):
//
//   Combined intra-block subgraphs (pre-resolved at setup time):
//     gen_intra_subgraphs        : (N_total_intra,) Vec<Int,5>
//     {tag,a0,a1,a2,a3}
//                                  tag=0 → proper torsion, tag=1 → improper
//                                  torsion
//     gen_intra_subgraph_offsets : (N_block_types,) Int
//     gen_intra_params           : (N_total_intra,) Vec<Real,5>
//                                  proper:   {k1,k2,k3,k4,offset}
//                                  improper: {k,delta,0,0,0}
//
//   Inter-block torsions (hash-table lookup at runtime):
//     gen_atom_type_hierarchy      : (N_block_types, max_atoms, MAX_HIER_DEPTH)
//     Int
//                                    per-atom type-index hierarchy
//     gen_connection_bond_types    : (N_block_types, max_n_conns) Int
//                                    bond-type int for each connection
//     gen_inter_torsion_hash_keys  : (N_hash_entries,) Vec<Int,6>
//                                    {t1,t2,t3,t4,bond_type,val_idx}
//     gen_inter_torsion_hash_values: (N_torsion_entries,) Vec<Real,5>
//     {k1,k2,k3,k4,off}
//
// Output (1 score type: gen_torsions, index 0):
//   V_t      : (1, n_poses, n_V, n_V)   where n_V = max_n_blocks or 1
//   dV_dx_t  : (1, n_atoms)
// ---------------------------------------------------------------------------
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct GenBondedPoseScoreDispatch {
  static auto forward(
      // Standard rotamer-layout params (identical to
      // CartBondedPoseScoreDispatch)
      TView<Vec<Real, 3>, 1, D> rot_coords,
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

      // Term-specific tensors from get_score_term_attributes()
      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
      // Combined intra-block subgraphs (proper + improper, tagged)
      TView<Vec<Int, 5>, 1, D> gen_intra_subgraphs,
      TView<Int, 1, D> gen_intra_subgraph_offsets,
      TView<Vec<Real, 5>, 1, D> gen_intra_params,
      // Inter-block torsions (hash table)
      TView<Vec<Int, 3>, 2, D> gen_atom_type_hierarchy,
      TView<Int, 2, D> gen_connection_bond_types,
      TView<Vec<Int, 6>, 1, D> gen_inter_torsion_hash_keys,
      TView<Vec<Real, 5>, 1, D> gen_inter_torsion_hash_values,

      bool output_block_pair_energies,
      bool compute_derivs)
      -> std::tuple<
          TPack<Real, 4, D>,         // V_t
          TPack<Vec<Real, 3>, 2, D>  // dV_dx_t
          >;

  static auto backward(
      TView<Vec<Real, 3>, 1, D> rot_coords,
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

      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
      TView<Vec<Int, 5>, 1, D> gen_intra_subgraphs,
      TView<Int, 1, D> gen_intra_subgraph_offsets,
      TView<Vec<Real, 5>, 1, D> gen_intra_params,
      TView<Vec<Int, 3>, 2, D> gen_atom_type_hierarchy,
      TView<Int, 2, D> gen_connection_bond_types,
      TView<Vec<Int, 6>, 1, D> gen_inter_torsion_hash_keys,
      TView<Vec<Real, 5>, 1, D> gen_inter_torsion_hash_values,

      TView<Real, 4, D> dTdV) -> TPack<Vec<Real, 3>, 2, D>;
};

// ---------------------------------------------------------------------------
// Rotamer-level scoring dispatch
// ---------------------------------------------------------------------------
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct GenBondedRotamerScoreDispatch {
  static auto forward(
      TView<Vec<Real, 3>, 1, D> rot_coords,
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

      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
      TView<Vec<Int, 5>, 1, D> gen_intra_subgraphs,
      TView<Int, 1, D> gen_intra_subgraph_offsets,
      TView<Vec<Real, 5>, 1, D> gen_intra_params,
      TView<Vec<Int, 3>, 2, D> gen_atom_type_hierarchy,
      TView<Int, 2, D> gen_connection_bond_types,
      TView<Vec<Int, 6>, 1, D> gen_inter_torsion_hash_keys,
      TView<Vec<Real, 5>, 1, D> gen_inter_torsion_hash_values,

      bool output_block_pair_energies,
      bool compute_derivs)
      -> std::tuple<
          TPack<Real, 2, D>,          // V_t
          TPack<Vec<Real, 3>, 2, D>,  // dV_dx_t
          TPack<Int, 2, D>,           // dispatch_indices_t
          TPack<Int, 1, D>,           // n_output_intxns_for_rot_conn_offset
          TPack<Int, 1, D>            // rotconn_for_output_intxn
          >;

  static auto backward(
      TView<Vec<Real, 3>, 1, D> rot_coords,
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

      TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
      TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
      TView<Vec<Int, 5>, 1, D> gen_intra_subgraphs,
      TView<Int, 1, D> gen_intra_subgraph_offsets,
      TView<Vec<Real, 5>, 1, D> gen_intra_params,
      TView<Vec<Int, 3>, 2, D> gen_atom_type_hierarchy,
      TView<Int, 2, D> gen_connection_bond_types,
      TView<Vec<Int, 6>, 1, D> gen_inter_torsion_hash_keys,
      TView<Vec<Real, 5>, 1, D> gen_inter_torsion_hash_values,

      TView<Int, 2, D> dispatch_indices,
      TView<Int, 1, D> n_output_intxns_for_rot_conn_offset,
      TView<Int, 1, D> rotconn_for_output_intxn,
      TView<Real, 2, D> dTdV) -> TPack<Vec<Real, 3>, 2, D>;
};

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
