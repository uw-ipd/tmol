#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>

#include <tmol/score/unresolved_atom.hh>
#include <tmol/score/common/diamond_macros.hh>

namespace tmol {
namespace score {
namespace common {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC int resolve_atom_from_uaid(
    UnresolvedAtomID<Int> uaid,
    int block_index,
    int pose_index,

    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn) {
  if (uaid.atom_id != -1) {  // This uaid resides in this block
    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][block_index];
    return block_coord_offset + uaid.atom_id;  // The actual global atom index
  } else if (uaid.conn_id != -1) {  // We need to follow to another block
    int connection_index = uaid.conn_id;
    int sep = uaid.n_bonds_from_conn;

    const Vec<Int, 2>& connection =
        pose_stack_inter_block_connections[pose_index][block_index]
                                          [connection_index];
    int other_block_index = connection[0];

    if (other_block_index == -1) {
      // This residue doesn't exist!
      return -1;
    }

    int other_connection_index = connection[1];
    int other_block_type_index =
        pose_stack_block_type[pose_index][other_block_index];

    int idx = block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_connection_index]
                                                [sep];  // The offset within the
                                                        // other block
    if (idx < 0) {
      return -1;
    }
    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][other_block_index];
    return block_coord_offset + idx;
  } else {
    // printf("uaid with both -1 for atom_id and conn_id\n");
    return -1;
  }
}

// Same logic: just return both the block index and the atom index
template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC auto resolve_local_atom_ind_from_uaid(
    UnresolvedAtomID<Int> uaid,
    int block_index,
    int pose_index,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn)
    -> std::tuple<Int, Int> {
  if (uaid.atom_id != -1) {  // This uaid resides in this block
    return {block_index, uaid.atom_id};
  } else if (uaid.conn_id != -1) {  // We need to follow to another block
    int connection_index = uaid.conn_id;
    int sep = uaid.n_bonds_from_conn;

    const Vec<Int, 2>& connection =
        pose_stack_inter_block_connections[pose_index][block_index]
                                          [connection_index];
    int other_block_index = connection[0];

    if (other_block_index == -1) {
      // This residue doesn't exist!
      return {-1, -1};
    }

    int other_connection_index = connection[1];
    int other_block_type_index =
        pose_stack_block_type[pose_index][other_block_index];

    int idx = block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_connection_index]
                                                [sep];  // The offset within the
                                                        // other block
    if (idx < 0) {
      return {-1, -1};
    }
    // int block_coord_offset =
    //     pose_stack_block_coord_offset[pose_index][other_block_index];
    return {other_block_index, idx};
  } else {
    // printf("uaid with both -1 for atom_id and conn_id\n");
    return {-1, -1};
  }
}

}  // namespace common
}  // namespace score
}  // namespace tmol
