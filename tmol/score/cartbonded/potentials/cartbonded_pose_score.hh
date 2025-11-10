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
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
struct CartBondedPoseScoreDispatch {
  static auto forward(
      // common params
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
      TView<Int, 2, D> atom_unique_ids,
      TView<Int, 2, D> atom_wildcard_ids,
      TView<Vec<Int, 5>, 1, D> hash_keys,
      TView<Vec<Real, 7>, 1, D> hash_values,
      TView<Vec<Int, 4>, 1, D> cart_subgraphs,

      // What is the index of the first intra-block subgraph for a block type
      // among all the subgraphs for all intra-block subgraphs?
      TView<Int, 1, D> cart_subgraph_offsets,

      // How many intra-block subgraphs of the three types (lengths, angles, &
      // torsions) are there?
      TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
      // What are the _local_ offsets for each of the three types; i.e.
      // relative to the offset listed in cart_subgraph_offsets, where
      // do the subgraphs for each of the three types begin?
      TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,

      // int max_subgraphs_per_block,
      bool output_block_pair_energies,

      bool compute_derivs

      )
      -> std::tuple<
          TPack<Real, 2, D>,          // V_t,
          TPack<Vec<Real, 3>, 2, D>,  // dV_dx_t,
          TPack<Int, 2, D>,           // dispatch_indices_t,
          TPack<Int, 1, D>,           // n_output_intxns_for_rot_conn_offset,
          TPack<Int, 1, D>            // rotconn_for_output_intxn,
          >;

  static auto backward(
      // common params
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
      TView<Int, 2, D> atom_unique_ids,
      TView<Int, 2, D> atom_wildcard_ids,
      TView<Vec<Int, 5>, 1, D> hash_keys,
      TView<Vec<Real, 7>, 1, D> hash_values,
      TView<Vec<Int, 4>, 1, D> cart_subgraphs,

      // What is the index of the first intra-block subgraph for a block type
      // among all the subgraphs for all intra-block subgraphs?
      TView<Int, 1, D> cart_subgraph_offsets,

      // How many intra-block subgraphs of the three types (lengths, angles, &
      // torsions) are there?
      TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
      // What are the _local_ offsets for each of the three types; i.e.
      // relative to the offset listed in cart_subgraph_offsets, where
      // do the subgraphs for each of the three types begin?
      TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,

      TView<Int, 2, D> dispatch_indices,
      TView<Int, 1, D> n_output_intxns_for_rot_conn_offset,
      TView<Int, 1, D> rotconn_for_output_intxn,

      TView<Real, 2, D> dTdV  // nterms x n-dispatch
      ) -> TPack<Vec<Real, 3>, 2, D>;
};

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
