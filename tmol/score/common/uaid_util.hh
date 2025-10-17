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
TMOL_DEVICE_FUNC auto resolve_local_atom_ind_from_uaid(
    UnresolvedAtomID<Int> uaid,
    int block_index,
    int pose_index,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn)
    -> std::tuple<Int, Int> {
  // Resolve an unresolved atom ID and return the block- and
  // atom-within-the-block indices

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
    return {other_block_index, idx};
  } else {
    return {-1, -1};
  }
}

template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC auto resolve_local_rotamer_atom_ind_from_uaid(
    UnresolvedAtomID<Int> uaid,
    int rotamer_index,
    int block_index,
    int pose_index,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn)
    -> std::tuple<Int, Int> {
  // Resolve an unresolved atom ID and return the block- and
  // atom-within-the-block indices

  if (uaid.atom_id != -1) {  // This uaid resides in this block/rotamer
    return {rotamer_index, uaid.atom_id};
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
    int other_rotamer_index = first_rot_for_block[pose_index][other_block_index];
    int other_block_type_index =
        first_rot_block_type[pose_index][other_block_index];

    int idx = block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_connection_index]
                                                [sep];  // The offset within the
                                                        // other block
    if (idx < 0) {
      return {-1, -1};
    }
    return {other_rotamer_index, idx};
  } else {
    return {-1, -1};
  }
}

template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC auto resolve_local_rotamer_pair_atom_ind_from_uaid(
    UnresolvedAtomID<Int> uaid,
    int rotamer_index1,
    int rotamer_index2,
    int block_index1, // <-- the source of the uaid
    int block_index2, // <-- the residue the other rotamer
    int block_type2, // <-- the block type of rotamer2
    int pose_index,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn)
    -> std::tuple<Int, Int> {
  // Resolve an unresolved atom ID and return the rotamer- and
  // atom-within-the-rotamer indices

  if (uaid.atom_id != -1) {  // This uaid resides in this block/rotamer
    return {rotamer_index1, uaid.atom_id};
  } else if (uaid.conn_id != -1) {  // We need to follow to another block
    int connection_index = uaid.conn_id;
    int sep = uaid.n_bonds_from_conn;

    const Vec<Int, 2>& connection =
        pose_stack_inter_block_connections[pose_index][block_index1]
                                          [connection_index];
    int other_block_index = connection[0];

    if (other_block_index == -1) {
      // This residue doesn't exist!
      return {-1, -1};
    }

    int other_connection_index = connection[1];

    int other_rotamer_index;
    int other_block_type_index;
    if (other_block_index == block_index2) {
      // Okay, this atom is in rotamer2
      other_rotamer_index = rotamer_index2;
      other_block_type_index = block_type2;
    } else {
      // This atom is not in rotamer1 or in rotamer2
      other_rotamer_index = first_rot_for_block[pose_index][other_block_index];
      other_block_type_index =
          first_rot_block_type[pose_index][other_block_index];
    }

    int idx = block_type_atom_downstream_of_conn[other_block_type_index]
                                                [other_connection_index]
                                                [sep];  // The offset within the
                                                        // other block
    if (idx < 0) {
      return {-1, -1};
    }
    return {other_rotamer_index, idx};
  } else {
    return {-1, -1};
  }
}

template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC int resolve_atom_from_uaid(
    UnresolvedAtomID<Int> uaid,
    int block_index,
    int pose_index,

    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn) {
  // Resolve an unresolved atom ID and return its pose-wide index
  auto resolved_atom_tuple = resolve_local_atom_ind_from_uaid(
      uaid,
      block_index,
      pose_index,
      pose_stack_block_coord_offset,
      pose_stack_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn);
  int resolved_atom_block_index = std::get<0>(resolved_atom_tuple);
  int resolved_atom_index = std::get<1>(resolved_atom_tuple);
  if (resolved_atom_block_index == -1 || resolved_atom_index == -1) {
    return -1;
  } else {
    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][resolved_atom_block_index];
    return block_coord_offset + resolved_atom_index;
  }
}

template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC int resolve_rotamer_atom_from_uaid(
    UnresolvedAtomID<Int> uaid,
    int rotamer_index,
    int block_index,
    int pose_index,

    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn) {
  // Resolve an unresolved atom ID and return its rotamer-set-wide index
  auto resolved_atom_tuple = resolve_local_rotamer_atom_ind_from_uaid(
      uaid,
      rotamer_index,
      block_index,
      pose_index,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn);
  int resolved_atom_rotamer_index = std::get<0>(resolved_atom_tuple);
  int resolved_atom_index = std::get<1>(resolved_atom_tuple);
  if (resolved_atom_rotamer_index == -1 || resolved_atom_index == -1) {
    return -1;
  } else {
    int rot_offset =
        rot_coord_offset[resolved_atom_rotamer_index];
    return rot_offset + resolved_atom_index;
  }
}

template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC int resolve_rotamer_pair_atom_from_uaid(
    UnresolvedAtomID<Int> uaid, // on block_index1
    int rotamer_index1,
    int rotamer_index2,
    int block_index1,
    int block_index2,
    int block_type2, // <-- the block type of rotamer2
    int pose_index,

    TView<Int, 1, D> rot_coord_offset,
    TView<Int, 2, D> first_rot_for_block,
    TView<Int, 2, D> first_rot_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> block_type_atom_downstream_of_conn) {
  // Resolve an unresolved atom ID and return its rotamer-set-wide index
  auto resolved_atom_tuple = resolve_local_rotamer_pair_atom_ind_from_uaid(
      uaid,
      rotamer_index1,
      rotamer_index2,
      block_index1,
      block_index2,
      block_type2,
      pose_index,
      first_rot_for_block,
      first_rot_block_type,
      pose_stack_inter_block_connections,
      block_type_atom_downstream_of_conn);
  int resolved_atom_rotamer_index = std::get<0>(resolved_atom_tuple);
  int resolved_atom_index = std::get<1>(resolved_atom_tuple);
  if (resolved_atom_rotamer_index == -1 || resolved_atom_index == -1) {
    return -1;
  } else {
    int rot_offset =
        rot_coord_offset[resolved_atom_rotamer_index];
    return rot_offset + resolved_atom_index;
  }
}

}  // namespace common
}  // namespace score
}  // namespace tmol
