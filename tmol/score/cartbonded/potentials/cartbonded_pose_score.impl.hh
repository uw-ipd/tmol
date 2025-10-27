#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/connection.hh>
#include <tmol/score/common/count_pair.hh>
#include <tmol/score/common/data_loading.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/hash_util.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/uaid_util.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include <tmol/score/cartbonded/potentials/cartbonded_pose_score.hh>

// Operator definitions; safe for CPU compilation
#include <moderngpu/operators.hxx>

#include <chrono>

#include "potentials.hh"

namespace tmol {
namespace score {
namespace cartbonded {
namespace potentials {

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;
template <typename Real>
using CoordQuad = Eigen::Matrix<Real, 4, 3>;
#define Real3 Vec<Real, 3>

enum class subgraph_type { length, angle, torsion };

template <typename Int>
TMOL_DEVICE_FUNC subgraph_type get_subgraph_type(Vec<Int, 4> subgraph) {
  if (subgraph[2] == -1 && subgraph[3] == -1) return subgraph_type::length;
  if (subgraph[3] == -1) return subgraph_type::angle;
  return subgraph_type::torsion;
}

template <typename Int, Int size>
TMOL_DEVICE_FUNC Vec<Int, size> atom_local_to_global_indices(
    Vec<Int, size> local_indices, Int offset) {
  Vec<Int, size> global_indices;
  for (int i = 0; i < size; i++) {
    if (local_indices[i] != -1)
      global_indices[i] = local_indices[i] + offset;
    else
      global_indices[i] = -1;
  }
  return global_indices;
}

// Get the atom ID for an atom index from a lookup table, preserving (-1)s
template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC Int
get_atom_id(TensorAccessor<Int, 1, D> atom_id_table, Int atom_index) {
  return (atom_index == -1) ? -1 : atom_id_table[atom_index];
}

// From a Vec of atom indices and a lookup table, return a new Vec with their
// IDs preserving (-1)s
template <typename Int, Int size, tmol::Device D>
TMOL_DEVICE_FUNC Vec<Int, size> get_atom_ids(
    TensorAccessor<Int, 1, D> atom_id_table, Vec<Int, size> atoms) {
  Vec<Int, size> atom_ids;
  for (int i = 0; i < size; i++) {
    atom_ids[i] = get_atom_id(atom_id_table, atoms[i]);
  }
  return atom_ids;
}

// Reverse the non-(-1) elements of a subgraph
template <typename Int>
TMOL_DEVICE_FUNC void reverse_subgraph(Vec<Int, 4>& subgraph) {
  subgraph_type type = get_subgraph_type(subgraph);
  Int temp;
  switch (type) {
    case subgraph_type::length:
      temp = subgraph[0];
      subgraph[0] = subgraph[1];
      subgraph[1] = temp;
      break;
    case subgraph_type::angle:
      temp = subgraph[0];
      subgraph[0] = subgraph[2];
      subgraph[2] = temp;
      break;
    case subgraph_type::torsion:
      temp = subgraph[0];
      subgraph[0] = subgraph[3];
      subgraph[3] = temp;
      temp = subgraph[1];
      subgraph[1] = subgraph[2];
      subgraph[2] = temp;
      break;
  }
}

template <typename Real, typename Int, int N, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_result(
    common::tuple<Real, Vec<Real3, N>> to_add,
    Vec<Int, N> atoms,
    Real& V,
    TensorAccessor<Vec<Real, 3>, 1, D> dV,
    const Real& weight = 1.0) {
  accumulate<D, Real>::add(V, common::get<0>(to_add));
  for (int i = 0; i < N; i++) {
    accumulate<D, Vec<Real, 3>>::add(
        dV[atoms[i]], common::get<1>(to_add)[i] * weight);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedPoseScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
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

    // How many intra-block subgraphs of the three types (lengths, angles, & torsions)
    // are there?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
    // What are the _local_ offsets for each of the three types; i.e.
    // relative to the offset listed in cart_subgraph_offsets, where 
    // do the subgraphs for each of the three types begin?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,

    // int max_subgraphs_per_block,
    bool output_block_pair_energies,

    bool compute_derivs

    ) -> std::tuple<
      TPack<Real, 2, D>,          // V_t,    
      TPack<Vec<Real, 3>, 2, D>,  // dV_dx_t,           
      TPack<Int, 2, D>,           // dispatch_indices_t,  
      TPack<Int, 1, D>,           // n_intxns_for_rot_conn_offset_t,  
      TPack<Int, 1, D>,           // rotconn_for_intxn_t,  
      TPack<Int, 1, D>,           // count_n_at_pair_dists_for_rotconn_offset_t,  
      TPack<Int, 1, D>,           // rotconn_for_lengths_t,  
      TPack<Int, 1, D>,           // count_n_at_trip_angls_for_rotconn_offset_t,  
      TPack<Int, 1, D>,           // rotconn_for_angles_t,  
      TPack<Int, 1, D>,           // count_n_at_quad_dihes_for_rotconn_offset,  
      TPack<Int, 1, D>            // rotconn_for_torsions_t                 
    > {
  using tmol::score::common::get_n_connection_spanning_subgraphs;
  using tmol::score::common::get_n_connection_spanning_subgraphs;
  using tmol::score::common::get_connection_spanning_subgraphs_offset;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_max_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = cart_subgraph_offsets.size(0);
  int const n_subgraphs = cart_subgraphs.size(0);
  int const n_max_atoms_per_block = atom_unique_ids.size(1);

  assert(pose_ind_for_atom.size(0) == n_atoms);
  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);
  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);
  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);

  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == n_max_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == n_max_conns);
  assert(atom_paths_from_conn.size(2) == MAX_PATHS_FROM_CONN);

  assert(atom_unique_ids.size(0) == n_block_types);
  assert(atom_unique_ids.size(1) == n_max_atoms_per_block);

  assert(atom_wildcard_ids.size(0) == n_block_types);
  assert(atom_wildcard_ids.size(1) == n_max_atoms_per_block);

  assert(cart_subgraph_offsets.size(0) == n_block_types);
  assert(cart_subgraph_type_counts.size(0) == n_block_types);
  assert(cart_subgraph_type_offsets.size(0) == n_block_types);

  // Algorithm:
  // 1. Count the number of rotamer-single and rotamer-pairs
  // we will need to dispatch over, and for each interaction, how many
  // bonds, angles, and torsions we'll calculate
  // 2. Then use load-balancing search for these three interactions
  // so we can assign one thread per atom-tuple
  // 3. Eval lenghts
  // 4. Eval angles
  // 5. Eval torsions


  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;


  // Convention: conn == n_max_conns ==> the intra-rotamer set of cart energies
  int const max_n_interactions = n_rots * (n_max_conns + 1);

  auto n_intxns_for_rot_conn_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  auto n_intxns_for_rot_conn = n_intxns_for_rot_conn_t.view;
  auto n_intxns_for_rot_conn_offset_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  auto n_intxns_for_rot_conn_offset = n_intxns_for_rot_conn_offset_t.view;


  auto count_n_at_pair_dists_for_rotconn_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  auto count_n_at_trip_angls_for_rotconn_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  auto count_n_at_quad_dihes_for_rotconn_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  auto count_n_at_pair_dists_for_rotconn_offset_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  auto count_n_at_trip_angls_for_rotconn_offset_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  auto count_n_at_quad_dihes_for_rotconn_offset_t = TPack<Int, 1, D>::zeros({max_n_interactions});
  
  auto count_n_at_pair_dists_for_rotconn = count_n_at_pair_dists_for_rotconn_t.view;
  auto count_n_at_trip_angls_for_rotconn = count_n_at_trip_angls_for_rotconn_t.view;
  auto count_n_at_quad_dihes_for_rotconn = count_n_at_quad_dihes_for_rotconn_t.view;
  auto count_n_at_pair_dists_for_rotconn_offset = count_n_at_pair_dists_for_rotconn_offset_t.view;
  auto count_n_at_trip_angls_for_rotconn_offset = count_n_at_trip_angls_for_rotconn_offset_t.view;
  auto count_n_at_quad_dihes_for_rotconn_offset = count_n_at_quad_dihes_for_rotconn_offset_t.view;

  auto count_intxns_for_rot_conn = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rot_ind = index / (n_max_conns + 1);
    int const conn_ind = index % (n_max_conns + 1);

    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    int const block_type_ind = block_type_ind_for_rot[rot_ind];

    bool const is_intra_conn = conn_ind == n_max_conns;

    if (is_intra_conn) {
      n_intxns_for_rot_conn[rot_ind * (n_max_conns + 1) + conn_ind] = 1;
      count_n_at_pair_dists_for_rotconn[index] = cart_subgraph_type_counts[block_type_ind][subgraph_length];
      count_n_at_trip_angls_for_rotconn[index] = cart_subgraph_type_counts[block_type_ind][subgraph_angle];
      count_n_at_quad_dihes_for_rotconn[index] = cart_subgraph_type_counts[block_type_ind][subgraph_torsion];
    } else {
      int const other_block_ind = pose_stack_inter_block_connections[
        pose_ind
      ][block_ind][conn_ind][0];
      int const other_conn_ind = pose_stack_inter_block_connections[
        pose_ind
      ][block_ind][conn_ind][1];
      // if (other_block_ind == -1) {
      //   return;
      // }
      if (other_block_ind < block_ind) {
        // Not an upper neighbor; therefore we will
        // count this interaction from the other direction
        // and should skip it from tihs directions
        // Because block_ind >= 0, this also handles
        // the case when other_block_ind == -1
        return;
      }
      int const other_block_n_rots = n_rots_for_block[pose_ind][other_block_ind];
      n_intxns_for_rot_conn[rot_ind * (n_max_conns + 1) + conn_ind] = other_block_n_rots;
      count_n_at_pair_dists_for_rotconn[index] = other_block_n_rots * get_n_connection_spanning_subgraphs(subgraph_length);
      count_n_at_trip_angls_for_rotconn[index] = other_block_n_rots * get_n_connection_spanning_subgraphs(subgraph_angle);
      count_n_at_quad_dihes_for_rotconn[index] = other_block_n_rots * get_n_connection_spanning_subgraphs(subgraph_torsion);
    }
  });
  // Launch this kernel for max_n_interactions threads
  DeviceDispatch<D>::template forall<launch_t>(
    max_n_interactions, count_intxns_for_rot_conn);

  // Scan and load-balancing sort on n intxns
  int n_intxns_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_intxns_for_rot_conn.data(),
          n_intxns_for_rot_conn_offset.data(),
          max_n_interactions,
          mgpu::plus_t<Int>());
  TPack<Int, 1, D> rotconn_for_intxn_t =
        DeviceDispatch<D>::template load_balancing_search<launch_t>(
          n_intxns_total,
          n_intxns_for_rot_conn_offset.data(),
          max_n_interactions
        );
  auto rotconn_for_intxn = rotconn_for_intxn_t.view;

  // Allocate the tensors to which we will write our outputs
  int const n_V = output_block_pair_energies ? n_intxns_total : n_poses;
  auto V_t = TPack<Real, 2, D>::zeros({5, n_V});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms});
  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_intxns_total});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;


  // Scan and load-balancing sort on distances
  int n_length_intxns_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          count_n_at_pair_dists_for_rotconn.data(),
          count_n_at_pair_dists_for_rotconn_offset.data(),
          max_n_interactions,
          mgpu::plus_t<Int>());
  TPack<Int, 1, D> rotconn_for_lengths_t =
        DeviceDispatch<D>::template load_balancing_search<launch_t>(
          n_length_intxns_total,
          count_n_at_pair_dists_for_rotconn_offset.data(),
          max_n_interactions
        );
  auto rotconn_for_lengths = rotconn_for_lengths_t.view;

  // Scan and load-balancing sort on angles
  int n_angle_intxns_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          count_n_at_trip_angls_for_rotconn.data(),
          count_n_at_trip_angls_for_rotconn_offset.data(),
          max_n_interactions,
          mgpu::plus_t<Int>());
  TPack<Int, 1, D> rotconn_for_angles_t =
        DeviceDispatch<D>::template load_balancing_search<launch_t>(
          n_angle_intxns_total,
          count_n_at_trip_angls_for_rotconn_offset.data(),
          max_n_interactions
        );
  auto rotconn_for_angles = rotconn_for_angles_t.view;

  // Scan and load-balancing sort on torsions
  int n_torsion_intxns_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          count_n_at_quad_dihes_for_rotconn.data(),
          count_n_at_quad_dihes_for_rotconn_offset.data(),
          max_n_interactions,
          mgpu::plus_t<Int>());
  TPack<Int, 1, D> rotconn_for_torsions_t =
        DeviceDispatch<D>::template load_balancing_search<launch_t>(
          n_torsion_intxns_total,
          count_n_at_quad_dihes_for_rotconn_offset.data(),
          max_n_interactions
        );
  auto rotconn_for_torsions = rotconn_for_angles_t.view;


  auto record_dispatch_indices_for_intxns = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rotconn_ind = rotconn_for_intxn[index];
    int const rot_ind1 = rotconn_ind / (n_max_conns + 1);
    int const conn_ind1 = rotconn_ind % (n_max_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    dispatch_indices[0][index] = pose_ind;
    dispatch_indices[1][index] = rot_ind1;

    int rot_ind2;
    if (conn_ind1 == n_max_conns) {
      // intra-residue:
      rot_ind2 = rot_ind1;
    } else {
      // inter-residue
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const rotconn_offset = n_intxns_for_rot_conn_offset[rotconn_ind];
      int const local_rot_ind2 = index - rotconn_offset;
      int const block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][0];
      rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
    }
    dispatch_indices[2][index] = rot_ind2;
  });
  // Record the rotamer pair indices for the interactions; though we will
  // only use them downstream if we are in output_block_pair_energies mode
  DeviceDispatch<D>::template forall<launch_t>(
    n_intxns_total, record_dispatch_indices_for_intxns);

  auto eval_lengths = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rotconn_ind = rotconn_for_lengths[index];
    int const rot_ind1 = rotconn_ind / (n_max_conns + 1);
    int const conn_ind1 = rotconn_ind % (n_max_conns + 1);
    int const rotconn_length_offset = count_n_at_pair_dists_for_rotconn_offset[rotconn_ind];
    int const local_length_ind_for_rotconn = index - rotconn_length_offset;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    int param_index = -1;
    Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

    int const rotconn_offset = n_intxns_for_rot_conn_offset[rotconn_ind];
    int dispatch_ind;

    if (conn_ind1 == n_max_conns) {
      // intra-residue!

      // There is only one interaction for this conn_ind1, so the
      // "dispatch index" is equal to the rotconn_offset.
      dispatch_ind = rotconn_offset;
      int const subgraph_offset = cart_subgraph_offsets[block_type1];
      // int const subgraph_offset_next = block_type1 + 1 == n_block_types
      //                               ? n_subgraphs
      //                               : cart_subgraph_offsets[block_type1 + 1];
      int const subgraph_ind = subgraph_offset + local_length_ind_for_rotconn;

      for (bool reverse : {false, true}) {
        Vec<Int, 4> subgraph = cart_subgraphs[subgraph_ind];
        if (reverse) reverse_subgraph(subgraph);

        Vec<Int, 4> subgraph_atom_ids =
            get_atom_ids(atom_unique_ids[block_type1], subgraph);
        param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

        subgraph_atom_indices = atom_local_to_global_indices(subgraph, rot_coord_offset1);
      }
    } else {
      // inter residue!
      int const block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][0];
      int const conn_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][1];

      // which rotamer on the other block?
      // local_length_ind_for_rotconn / n_lengths, but n_lengths == 1
      // so let's skip the math.
      int const local_rot_ind2 = local_length_ind_for_rotconn;

      // The "dispatch index" is rotconn_offset + the local index of the
      // block2 rotamer
      dispatch_ind = rotconn_offset + local_rot_ind2;

      int const local_subgraph_ind = 0; // only one length and its the first one
      int const subgraph_offset = get_connection_spanning_subgraphs_offset(subgraph_length);
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // this is a long way to go to add 0 to 0!
      int const subgraph_index = local_subgraph_ind + subgraph_offset;

      tuple<int, int, int> spanning_subgraphs =
          get_connection_spanning_subgraph_indices(subgraph_index);
      int res1_path_ind = common::get<1>(spanning_subgraphs);
      int res2_path_ind = common::get<2>(spanning_subgraphs);

      // Grab the paths from each block
      Vec<Int, 3> res1_path =
          atom_paths_from_conn[block_type1][conn_ind1][res1_path_ind];
      Vec<Int, 3> res2_path =
          atom_paths_from_conn[block_type2][conn_ind2][res2_path_ind];
      // Make sure these are valid paths
      if (res1_path[0] == -1 || res2_path[0] == -1) return;
      // Reverse the first path so that we can join them head-to-head
      res1_path.reverseInPlace();
      // Get a new Vec containing the global indices of the atoms
      Vec<Int, 3> res1_atom_indices =
          atom_local_to_global_indices(res1_path, rot_coord_offset1);
      Vec<Int, 3> res2_atom_indices =
          atom_local_to_global_indices(res2_path, rot_coord_offset2);
      // Calculate the size of each path
      Int res1_size = 1; // (res1_atom_indices.array() != -1).count();
      Int res2_size = 1; // (res2_atom_indices.array() != -1).count();

      // Try both unique and wildcard IDs for block 1
      for (bool wildcard : {false, true}) {
        // Get the lookup tables for atom ID
        const auto& res1_atom_id_table = (wildcard)
                                              ? atom_wildcard_ids[block_type1]
                                              : atom_unique_ids[block_type1];
        const auto& res2_atom_id_table = atom_wildcard_ids[block_type2];

        // Get the atom IDs
        Vec<Int, 3> res1_subgraph_atom_ids =
            get_atom_ids(res1_atom_id_table, res1_path);
        Vec<Int, 3> res2_subgraph_atom_ids =
            get_atom_ids(res2_atom_id_table, res2_path);

        // Make the joined data structures
        Vec<Int, 4> path;
        Vec<Int, 4> atom_indices;
        Vec<Int, 4> subgraph_atom_ids;

        // Init with -1s
        path << -1, -1, -1, -1;
        atom_indices << -1, -1, -1, -1;
        subgraph_atom_ids << -1, -1, -1, -1;

        // Join the paths into 1
        path.head(res1_size + res2_size) << res1_path.tail(res1_size),
            res2_path.head(res2_size);
        atom_indices.head(res1_size + res2_size)
            << res1_atom_indices.tail(res1_size),
            res2_atom_indices.head(res2_size);
        subgraph_atom_ids.head(res1_size + res2_size)
            << res1_subgraph_atom_ids.tail(res1_size),
            res2_subgraph_atom_ids.head(res2_size);

        // Do the lookup
        int param_index =
            hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
        if (param_index != -1) {
          // we found it!
          subgraph_atom_indices = atom_indices;
          break;
        }
      }
    }

    if (param_index != -1) {
      // score_subgraph(subgraph_atom_indices, param_index);
      Vec<Real, 7> params = hash_values[param_index];
      Vec<Real, 3> atom1 = rot_coords[subgraph_atom_indices[0]];
      Vec<Real, 3> atom2 = rot_coords[subgraph_atom_indices[1]];

      int score_type = params[0];

      int V_ind = (output_block_pair_energies) ? dispatch_ind : pose_ind;

      auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
      accumulate_result<Real, Int, 2, D>(
          eval,
          subgraph_atom_indices.head(2),
          V[score_type][V_ind],
          dV_dx[score_type],
          1.0);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
    n_length_intxns_total, eval_lengths);


  auto eval_angles = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rotconn_ind = rotconn_for_angles[index];
    int const rot_ind1 = rotconn_ind / (n_max_conns + 1);
    int const conn_ind1 = rotconn_ind % (n_max_conns + 1);
    int const rotconn_angle_offset = count_n_at_trip_angls_for_rotconn_offset[rotconn_ind];
    int const local_rot_and_angle_ind_for_rotconn = index - rotconn_angle_offset;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    int param_index = -1;
    Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

    int const rotconn_offset = n_intxns_for_rot_conn_offset[rotconn_ind];
    int dispatch_ind;

    if (conn_ind1 == n_max_conns) {
      // intra-residue!

      // There is only one interaction for this conn_ind1, so the
      // "dispatch index" is equal to the rotconn_offset.
      dispatch_ind = rotconn_offset;

      // Get the subgraph for this particular angle we are looking at
      int const subgraph_offset = cart_subgraph_offsets[block_type1] +
        cart_subgraph_type_offsets[block_type1][subgraph_angle];
      int const subgraph_ind = subgraph_offset + local_rot_and_angle_ind_for_rotconn;

      for (bool reverse : {false, true}) {
        Vec<Int, 4> subgraph = cart_subgraphs[subgraph_ind];
        if (reverse) reverse_subgraph(subgraph);

        Vec<Int, 4> subgraph_atom_ids =
            get_atom_ids(atom_unique_ids[block_type1], subgraph);
        param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

        subgraph_atom_indices = atom_local_to_global_indices(subgraph, rot_coord_offset1);

        if (param_index != -1){
          break;
        }
      }
    } else {
      // inter residue!
      int const block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][0];
      int const conn_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][1];

      // which rotamer on the other block?
      // local_length_ind_for_rotconn / n_angles
      int const n_intra_angles = get_n_connection_spanning_subgraphs(subgraph_angle);
      int const local_rot_ind2 = local_rot_and_angle_ind_for_rotconn / n_intra_angles;
      int const local_angle_ind = local_rot_and_angle_ind_for_rotconn % n_intra_angles;

      // The "dispatch index" is rotconn_offset + the local index of the
      // block2 rotamer
      dispatch_ind = rotconn_offset + local_rot_ind2;

      // which angle for this rotconn?

      int const subgraph_offset = get_connection_spanning_subgraphs_offset(subgraph_angle);
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // the index of this subgraph in the list of inter-block subgraphs
      // from connection.hh. We do not know if this angle is defined
      // for this pair of block types yet, so we will attempt to resolve
      // the atom indices, and score only if the resolution succeeds
      int const subgraph_index = subgraph_offset + local_angle_ind;

      tuple<int, int, int> spanning_subgraphs =
          get_connection_spanning_subgraph_indices(subgraph_index);
      int res1_path_ind = common::get<1>(spanning_subgraphs);
      int res2_path_ind = common::get<2>(spanning_subgraphs);

      // Grab the paths from each block
      Vec<Int, 3> res1_path =
          atom_paths_from_conn[block_type1][conn_ind1][res1_path_ind];
      Vec<Int, 3> res2_path =
          atom_paths_from_conn[block_type2][conn_ind2][res2_path_ind];

      // Now make sure these are valid paths
      if (res1_path[0] == -1 || res2_path[0] == -1) return;
      // Reverse the first path so that we can join them head-to-head
      res1_path.reverseInPlace();
      // Get a new Vec containing the global indices of the atoms
      Vec<Int, 3> res1_atom_indices =
          atom_local_to_global_indices(res1_path, rot_coord_offset1);
      Vec<Int, 3> res2_atom_indices =
          atom_local_to_global_indices(res2_path, rot_coord_offset2);
      // Calculate the size of each path
      Int res1_size = 1; // (res1_atom_indices.array() != -1).count();
      Int res2_size = 1; // (res2_atom_indices.array() != -1).count();

      // Try both unique and wildcard IDs for block 1
      for (bool wildcard : {false, true}) {
        // Get the lookup tables for atom ID
        const auto& res1_atom_id_table = (wildcard)
                                              ? atom_wildcard_ids[block_type1]
                                              : atom_unique_ids[block_type1];
        const auto& res2_atom_id_table = atom_wildcard_ids[block_type2];

        // Get the atom IDs
        Vec<Int, 3> res1_subgraph_atom_ids =
            get_atom_ids(res1_atom_id_table, res1_path);
        Vec<Int, 3> res2_subgraph_atom_ids =
            get_atom_ids(res2_atom_id_table, res2_path);

        // Make the joined data structures
        Vec<Int, 4> path;
        Vec<Int, 4> atom_indices;
        Vec<Int, 4> subgraph_atom_ids;

        // Init with -1s
        path << -1, -1, -1, -1;
        atom_indices << -1, -1, -1, -1;
        subgraph_atom_ids << -1, -1, -1, -1;

        // Join the paths into 1
        path.head(res1_size + res2_size) << res1_path.tail(res1_size),
            res2_path.head(res2_size);
        atom_indices.head(res1_size + res2_size)
            << res1_atom_indices.tail(res1_size),
            res2_atom_indices.head(res2_size);
        subgraph_atom_ids.head(res1_size + res2_size)
            << res1_subgraph_atom_ids.tail(res1_size),
            res2_subgraph_atom_ids.head(res2_size);

        // Do the lookup
        int param_index =
            hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
        if (param_index != -1) {
          // we found it!
          subgraph_atom_indices = atom_indices;
          break;
        }
      }
    }

    if (param_index != -1) {
      // score_subgraph(subgraph_atom_indices, param_index);
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[subgraph_atom_indices[0]];
      Vec<Real, 3> atom2 = rot_coords[subgraph_atom_indices[1]];
      Vec<Real, 3> atom3 = rot_coords[subgraph_atom_indices[2]];

      int score_type = params[0];

      int V_ind = (output_block_pair_energies) ? dispatch_ind : pose_ind;

      auto eval =
          cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
      accumulate_result<Real, Int, 3, D>(
          eval,
          subgraph_atom_indices.head(3),
          V[score_type][V_ind],
          dV_dx[score_type],
          1.0);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
    n_angle_intxns_total, eval_angles);


  // And finally, torsions!
  auto eval_torsions = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rotconn_ind = rotconn_for_torsions[index];
    int const rot_ind1 = rotconn_ind / (n_max_conns + 1);
    int const conn_ind1 = rotconn_ind % (n_max_conns + 1);
    int const rotconn_torsion_offset = count_n_at_quad_dihes_for_rotconn_offset[rotconn_ind];
    int const local_rot_and_torsion_ind_for_rotconn = index - rotconn_torsion_offset;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    int param_index = -1;
    Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

    int const rotconn_offset = n_intxns_for_rot_conn_offset[rotconn_ind];
    int dispatch_ind;

    if (conn_ind1 == n_max_conns) {
      // intra-residue!

      // There is only one interaction for this conn_ind1, so the
      // "dispatch index" is equal to the rotconn_offset.
      dispatch_ind = rotconn_offset;

      // Get the subgraph for this particular angle we are looking at
      int const subgraph_offset = cart_subgraph_offsets[block_type1] +
        cart_subgraph_type_offsets[block_type1][subgraph_torsion];
      int const subgraph_ind = subgraph_offset + local_rot_and_torsion_ind_for_rotconn;

      for (bool reverse : {false, true}) {
        Vec<Int, 4> subgraph = cart_subgraphs[subgraph_ind];
        if (reverse) reverse_subgraph(subgraph);

        Vec<Int, 4> subgraph_atom_ids =
            get_atom_ids(atom_unique_ids[block_type1], subgraph);
        param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

        subgraph_atom_indices = atom_local_to_global_indices(subgraph, rot_coord_offset1);

        if (param_index != -1){
          break;
        }
      }
    } else {
      // inter residue!
      int const block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][0];
      int const conn_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][1];

      // which rotamer on the other block?
      // which torsion for this rotconn?
      int const n_intra_torsions = get_n_connection_spanning_subgraphs(subgraph_torsion);
      int const local_rot_ind2 = local_rot_and_torsion_ind_for_rotconn / n_intra_torsions;
      int const local_torsion_ind = local_rot_and_torsion_ind_for_rotconn % n_intra_torsions;

      // The "dispatch index" is rotconn_offset + the local index of the
      // block2 rotamer
      dispatch_ind = rotconn_offset + local_rot_ind2;


      int const subgraph_offset = get_connection_spanning_subgraphs_offset(subgraph_torsion);
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // the index of this subgraph in the list of inter-block subgraphs
      // from connection.hh. We do not know if this angle is defined
      // for this pair of block types yet, so we will attempt to resolve
      // the atom indices, and score only if the resolution succeeds
      int const subgraph_index = subgraph_offset + local_torsion_ind;

      tuple<int, int, int> spanning_subgraphs =
          get_connection_spanning_subgraph_indices(subgraph_index);
      int res1_path_ind = common::get<1>(spanning_subgraphs);
      int res2_path_ind = common::get<2>(spanning_subgraphs);

      // Grab the paths from each block
      Vec<Int, 3> res1_path =
          atom_paths_from_conn[block_type1][conn_ind1][res1_path_ind];
      Vec<Int, 3> res2_path =
          atom_paths_from_conn[block_type2][conn_ind2][res2_path_ind];

      // Now make sure these are valid paths
      if (res1_path[0] == -1 || res2_path[0] == -1) return;
      // Reverse the first path so that we can join them head-to-head
      res1_path.reverseInPlace();
      // Get a new Vec containing the global indices of the atoms
      Vec<Int, 3> res1_atom_indices =
          atom_local_to_global_indices(res1_path, rot_coord_offset1);
      Vec<Int, 3> res2_atom_indices =
          atom_local_to_global_indices(res2_path, rot_coord_offset2);
      // Calculate the size of each path
      Int res1_size = 1; // (res1_atom_indices.array() != -1).count();
      Int res2_size = 1; // (res2_atom_indices.array() != -1).count();

      // Try both unique and wildcard IDs for block 1
      for (bool wildcard : {false, true}) {
        // Get the lookup tables for atom ID
        const auto& res1_atom_id_table = (wildcard)
                                              ? atom_wildcard_ids[block_type1]
                                              : atom_unique_ids[block_type1];
        const auto& res2_atom_id_table = atom_wildcard_ids[block_type2];

        // Get the atom IDs
        Vec<Int, 3> res1_subgraph_atom_ids =
            get_atom_ids(res1_atom_id_table, res1_path);
        Vec<Int, 3> res2_subgraph_atom_ids =
            get_atom_ids(res2_atom_id_table, res2_path);

        // Make the joined data structures
        Vec<Int, 4> path;
        Vec<Int, 4> atom_indices;
        Vec<Int, 4> subgraph_atom_ids;

        // Init with -1s
        path << -1, -1, -1, -1;
        atom_indices << -1, -1, -1, -1;
        subgraph_atom_ids << -1, -1, -1, -1;

        // Join the paths into 1
        path.head(res1_size + res2_size) << res1_path.tail(res1_size),
            res2_path.head(res2_size);
        atom_indices.head(res1_size + res2_size)
            << res1_atom_indices.tail(res1_size),
            res2_atom_indices.head(res2_size);
        subgraph_atom_ids.head(res1_size + res2_size)
            << res1_subgraph_atom_ids.tail(res1_size),
            res2_subgraph_atom_ids.head(res2_size);

        // Do the lookup
        int param_index =
            hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
        if (param_index != -1) {
          // we found it!
          subgraph_atom_indices = atom_indices;
          break;
        }
      }
    }

    if (param_index != -1) {
      // score_subgraph(subgraph_atom_indices, param_index);
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[subgraph_atom_indices[0]];
      Vec<Real, 3> atom2 = rot_coords[subgraph_atom_indices[1]];
      Vec<Real, 3> atom3 = rot_coords[subgraph_atom_indices[2]];
      Vec<Real, 3> atom4 = rot_coords[subgraph_atom_indices[3]];

      int score_type = params[0];

      int V_ind = (output_block_pair_energies) ? dispatch_ind : pose_ind;

      auto eval = cbtorsion_V_dV(
          atom1,
          atom2,
          atom3,
          atom4,
          params[1],
          params[2],
          params[3],
          params[4],
          params[5],
          params[6]);
      accumulate_result<Real, Int, 4, D>(
          eval,
          subgraph_atom_indices.head(4),
          V[score_type][V_ind],
          dV_dx[score_type],
          1.0);
    }

  });
  DeviceDispatch<D>::template forall<launch_t>(
    n_torsion_intxns_total, eval_torsions);

  return {
    V_t,
    dV_dx_t,
    dispatch_indices_t,
    n_intxns_for_rot_conn_offset_t,
    rotconn_for_intxn_t,
    count_n_at_pair_dists_for_rotconn_offset_t,
    rotconn_for_lengths_t,
    count_n_at_trip_angls_for_rotconn_offset_t,
    rotconn_for_angles_t,
    count_n_at_quad_dihes_for_rotconn_offset_t,
    rotconn_for_torsions_t
  };

  // ///////////////////////////// OLD CODE /////////////////////////////////

  // // auto V_t = TPack<Real, 2, D>::zeros({5, n_poses});
  // //  auto V_t = TPack<Real, 2, D>::zeros({1, n_poses});
  // TPack<Real, 4, D> V_t;
  // if (output_block_pair_energies) {
  //   V_t = TPack<Real, 4, D>::zeros({5, n_poses, n_blocks, n_blocks});
  // } else {
  //   V_t = TPack<Real, 4, D>::zeros({5, n_poses, 1, 1});
  // }

  // auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({5, n_poses, n_max_atoms});

  // auto V = V_t.view;
  // auto dV_dx = dV_dx_t.view;

  // max_subgraphs_per_block +=
  //     NUM_INTER_RES_PATHS;  // Add in the inter-residue subgraphs

  // // Optimal launch box on v100 and a100 is nt=32, vt=1
  // LAUNCH_BOX_32;

  // auto func = ([=] TMOL_DEVICE_FUNC(
  //                  int pose_index, int block_index, int subgraph_index) {
  //   Real score = 0;

  //   int block_type = pose_stack_block_type[pose_index][block_index];
  //   auto pose_coords = coords[pose_index];
  //   int block_coord_offset =
  //       pose_stack_block_coord_offset[pose_index][block_index];
  //   if (block_type < 0) {
  //     return;
  //   }
  //   int subgraph_offset = cart_subgraph_offsets[block_type];
  //   int subgraph_offset_next = block_type + 1 == n_block_types
  //                                  ? n_subgraphs
  //                                  : cart_subgraph_offsets[block_type + 1];
  //   subgraph_index += subgraph_offset;

  //   auto score_subgraph =
  //       ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
  //         Vec<Real, 7> params = hash_values[param_index];

  //         Vec<Real, 3> atom1 = pose_coords[atoms[0]];
  //         Vec<Real, 3> atom2 = pose_coords[atoms[1]];

  //         subgraph_type type = get_subgraph_type(atoms);

  //         int score_type = params[0];

  //         int block_index_v = (output_block_pair_energies) ? block_index : 0;

  //         // Real score;
  //         switch (type) {
  //           case subgraph_type::length: {
  //             auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
  //             accumulate_result<Real, Int, 2, D>(
  //                 eval,
  //                 atoms.head(2),
  //                 V[score_type][pose_index][block_index_v][block_index_v],
  //                 dV_dx[score_type][pose_index],
  //                 1.0);

  //             break;
  //           }
  //           case subgraph_type::angle: {
  //             Vec<Real, 3> atom3 = pose_coords[atoms[2]];
  //             auto eval =
  //                 cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
  //             accumulate_result<Real, Int, 3, D>(
  //                 eval,
  //                 atoms.head(3),
  //                 V[score_type][pose_index][block_index_v][block_index_v],
  //                 dV_dx[score_type][pose_index],
  //                 1.0);

  //             break;
  //           }
  //           case subgraph_type::torsion: {
  //             Vec<Real, 3> atom3 = pose_coords[atoms[2]];
  //             Vec<Real, 3> atom4 = pose_coords[atoms[3]];
  //             auto eval = cbtorsion_V_dV(
  //                 atom1,
  //                 atom2,
  //                 atom3,
  //                 atom4,
  //                 params[1],
  //                 params[2],
  //                 params[3],
  //                 params[4],
  //                 params[5],
  //                 params[6]);
  //             accumulate_result<Real, Int, 4, D>(
  //                 eval,
  //                 atoms.head(4),
  //                 V[score_type][pose_index][block_index_v][block_index_v],
  //                 dV_dx[score_type][pose_index],
  //                 1.0);

  //             break;
  //           }
  //         }
  //       });

  //   if (subgraph_index >= subgraph_offset_next) {
  //     if (subgraph_index >= subgraph_offset_next + NUM_INTER_RES_PATHS) return;

  //     subgraph_index -= subgraph_offset_next;

  //     // Iterate over each connection in this block. We don't need to worry
  //     // about the other block accidentally duplicating the energy attribution
  //     // since the ordering of the atoms matters.
  //     for (int i = 0; i < n_max_conns; i++) {
  //       const Vec<Int, 2>& connection =
  //           pose_stack_inter_block_connections[pose_index][block_index][i];
  //       int other_block_index = connection[0];
  //       // No block on the other side of the connection, nothing to do
  //       if (other_block_index == -1) continue;
  //       int other_block_type =
  //           pose_stack_block_type[pose_index][other_block_index];
  //       int other_block_offset =
  //           pose_stack_block_coord_offset[pose_index][other_block_index];

  //       int other_connection_index = connection[1];

  //       // From our subgraph index, grab the corresponding paths indices for
  //       // each block
  //       tuple<int, int> spanning_subgraphs =
  //           get_connection_spanning_subgraph_indices(subgraph_index);
  //       int res1_path_ind = common::get<0>(spanning_subgraphs);
  //       int res2_path_ind = common::get<1>(spanning_subgraphs);

  //       // Grab the paths from each block
  //       Vec<Int, 3> res1_path =
  //           atom_paths_from_conn[block_type][i][res1_path_ind];
  //       Vec<Int, 3> res2_path =
  //           atom_paths_from_conn[other_block_type][other_connection_index]
  //                               [res2_path_ind];

  //       // Make sure these are valid paths
  //       if (res1_path[0] == -1 || res2_path[0] == -1) continue;
  //       // Reverse the first path so that we can join them head-to-head
  //       res1_path.reverseInPlace();

  //       // Get a new Vec containing the global indices of the atoms
  //       Vec<Int, 3> res1_atom_indices =
  //           atom_local_to_global_indices(res1_path, block_coord_offset);
  //       Vec<Int, 3> res2_atom_indices =
  //           atom_local_to_global_indices(res2_path, other_block_offset);

  //       // Calculate the size of each path
  //       Int res1_size = (res1_atom_indices.array() != -1).count();
  //       Int res2_size = (res2_atom_indices.array() != -1).count();

  //       // Try both unique and wildcard IDs for block 1
  //       for (bool wildcard : {false, true}) {
  //         // Get the lookup tables for atom ID
  //         const auto& res1_atom_id_table = (wildcard)
  //                                              ? atom_wildcard_ids[block_type]
  //                                              : atom_unique_ids[block_type];
  //         const auto& res2_atom_id_table = atom_wildcard_ids[other_block_type];

  //         // Get the atom IDs
  //         Vec<Int, 3> res1_subgraph_atom_ids =
  //             get_atom_ids(res1_atom_id_table, res1_path);
  //         Vec<Int, 3> res2_subgraph_atom_ids =
  //             get_atom_ids(res2_atom_id_table, res2_path);

  //         // Make the joined data structures
  //         Vec<Int, 4> path;
  //         Vec<Int, 4> atom_indices;
  //         Vec<Int, 4> subgraph_atom_ids;

  //         // Init with -1s
  //         path << -1, -1, -1, -1;
  //         atom_indices << -1, -1, -1, -1;
  //         subgraph_atom_ids << -1, -1, -1, -1;

  //         // Join the paths into 1
  //         path.head(res1_size + res2_size) << res1_path.tail(res1_size),
  //             res2_path.head(res2_size);
  //         atom_indices.head(res1_size + res2_size)
  //             << res1_atom_indices.tail(res1_size),
  //             res2_atom_indices.head(res2_size);
  //         subgraph_atom_ids.head(res1_size + res2_size)
  //             << res1_subgraph_atom_ids.tail(res1_size),
  //             res2_subgraph_atom_ids.head(res2_size);

  //         // Do the lookup
  //         int param_index =
  //             hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

  //         // If we found a param that matches, score the subgraph
  //         if (param_index != -1) {
  //           score_subgraph(atom_indices, param_index);
  //         }
  //       }
  //     }
  //     return;
  //   }

  //   // Intra-res subgraphs
  //   for (bool reverse : {false, true}) {
  //     Vec<Int, 4> subgraph = cart_subgraphs[subgraph_index];
  //     if (reverse) reverse_subgraph(subgraph);

  //     Vec<Int, 4> subgraph_atom_ids =
  //         get_atom_ids(atom_unique_ids[block_type], subgraph);
  //     int param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

  //     Vec<Int, 4> subgraph_atom_indices =
  //         atom_local_to_global_indices(subgraph, block_coord_offset);

  //     if (param_index != -1) {
  //       score_subgraph(subgraph_atom_indices, param_index);
  //     }
  //   }
  // });

  // DeviceDispatch<D>::foreach_combination_triple(
  //     n_poses, n_blocks, max_subgraphs_per_block, func);

  // return {V_t, dV_dx_t};
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedPoseScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
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

    // How many intra-block subgraphs of the three types (lengths, angles, & torsions)
    // are there?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
    // What are the _local_ offsets for each of the three types; i.e.
    // relative to the offset listed in cart_subgraph_offsets, where 
    // do the subgraphs for each of the three types begin?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,

    TView<Int, 2, D> dispatch_indices,
    TView<Int, 1, D> n_intxns_for_rot_conn_offset,
    TView<Int, 1, D> rotconn_for_intxn,
    TView<Int, 1, D> count_n_at_pair_dists_for_rotconn_offset,
    TView<Int, 1, D> rotconn_for_lengths,
    TView<Int, 1, D> count_n_at_trip_angls_for_rotconn_offset,
    TView<Int, 1, D> rotconn_for_angles,
    TView<Int, 1, D> count_n_at_quad_dihes_for_rotconn_offset,
    TView<Int, 1, D> rotconn_for_torsions,

    TView<Real, 2, D> dTdV              // nterms x n-dispatch
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const n_max_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = cart_subgraph_offsets.size(0);
  int const n_subgraphs = cart_subgraphs.size(0);
  int const n_max_atoms_per_block = atom_unique_ids.size(1);

  assert(pose_ind_for_atom.size(0) == n_atoms);
  assert(first_rot_block_type.size(0) == n_poses);
  assert(first_rot_block_type.size(1) == max_n_blocks);

  assert(block_ind_for_rot.size(0) == n_rots);
  assert(pose_ind_for_rot.size(0) == n_rots);
  assert(block_type_ind_for_rot.size(0) == n_rots);
  assert(n_rots_for_pose.size(0) == n_poses);
  assert(rot_offset_for_pose.size(0) == n_poses);

  assert(n_rots_for_block.size(0) == n_poses);
  assert(n_rots_for_block.size(1) == max_n_blocks);
  assert(rot_offset_for_block.size(0) == n_poses);
  assert(rot_offset_for_block.size(1) == max_n_blocks);
  
  assert(pose_stack_inter_block_connections.size(0) == n_poses);
  assert(pose_stack_inter_block_connections.size(1) == max_n_blocks);
  assert(pose_stack_inter_block_connections.size(2) == n_max_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == n_max_conns);
  assert(atom_paths_from_conn.size(2) == MAX_PATHS_FROM_CONN);

  assert(atom_unique_ids.size(0) == n_block_types);
  assert(atom_unique_ids.size(1) == n_max_atoms_per_block);

  assert(atom_wildcard_ids.size(0) == n_block_types);
  assert(atom_wildcard_ids.size(1) == n_max_atoms_per_block);

  assert(cart_subgraph_offsets.size(0) == n_block_types);
  assert(cart_subgraph_type_counts.size(0) == n_block_types);
  assert(cart_subgraph_type_offsets.size(0) == n_block_types);

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  int const n_intxns_total = dispatch_indices.size(1);
  // We only call backward if we are in output_block_pair_energies mode,
  // so we will just go directly to allocating V_t w/ n_intxns_total size
  // int const n_V = output_block_pair_energies ? n_intxns_total : n_poses;
  auto V_t = TPack<Real, 2, D>::zeros({5, n_intxns_total});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms});
  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;


  auto eval_lengths = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rotconn_ind = rotconn_for_lengths[index];
    int const rot_ind1 = rotconn_ind / (n_max_conns + 1);
    int const conn_ind1 = rotconn_ind % (n_max_conns + 1);
    int const rotconn_length_offset = count_n_at_pair_dists_for_rotconn_offset[rotconn_ind];
    int const local_length_ind_for_rotconn = index - rotconn_length_offset;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    int param_index = -1;
    Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

    int const rotconn_offset = n_intxns_for_rot_conn_offset[rotconn_ind];
    int dispatch_ind;

    if (conn_ind1 == n_max_conns) {
      // intra-residue!

      // There is only one interaction for this conn_ind1, so the
      // "dispatch index" is equal to the rotconn_offset.
      dispatch_ind = rotconn_offset;
      int const subgraph_offset = cart_subgraph_offsets[block_type1];
      // int const subgraph_offset_next = block_type1 + 1 == n_block_types
      //                               ? n_subgraphs
      //                               : cart_subgraph_offsets[block_type1 + 1];
      int const subgraph_ind = subgraph_offset + local_length_ind_for_rotconn;

      for (bool reverse : {false, true}) {
        Vec<Int, 4> subgraph = cart_subgraphs[subgraph_ind];
        if (reverse) reverse_subgraph(subgraph);

        Vec<Int, 4> subgraph_atom_ids =
            get_atom_ids(atom_unique_ids[block_type1], subgraph);
        param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

        subgraph_atom_indices = atom_local_to_global_indices(subgraph, rot_coord_offset1);
      }
    } else {
      // inter residue!
      int const block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][0];
      int const conn_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][1];

      // which rotamer on the other block?
      // local_length_ind_for_rotconn / n_lengths, but n_lengths == 1
      // so let's skip the math.
      int const local_rot_ind2 = local_length_ind_for_rotconn;

      // The "dispatch index" is rotconn_offset + the local index of the
      // block2 rotamer
      dispatch_ind = rotconn_offset + local_rot_ind2;

      int const local_subgraph_ind = 0; // only one length and its the first one
      int const subgraph_offset = get_connection_spanning_subgraphs_offset(subgraph_length);
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // this is a long way to go to add 0 to 0!
      int const subgraph_index = local_subgraph_ind + subgraph_offset;

      tuple<int, int, int> spanning_subgraphs =
          get_connection_spanning_subgraph_indices(subgraph_index);
      int res1_path_ind = common::get<1>(spanning_subgraphs);
      int res2_path_ind = common::get<2>(spanning_subgraphs);

      // Grab the paths from each block
      Vec<Int, 3> res1_path =
          atom_paths_from_conn[block_type1][conn_ind1][res1_path_ind];
      Vec<Int, 3> res2_path =
          atom_paths_from_conn[block_type2][conn_ind2][res2_path_ind];
      // Make sure these are valid paths
      if (res1_path[0] == -1 || res2_path[0] == -1) return;
      // Reverse the first path so that we can join them head-to-head
      res1_path.reverseInPlace();
      // Get a new Vec containing the global indices of the atoms
      Vec<Int, 3> res1_atom_indices =
          atom_local_to_global_indices(res1_path, rot_coord_offset1);
      Vec<Int, 3> res2_atom_indices =
          atom_local_to_global_indices(res2_path, rot_coord_offset2);
      // Calculate the size of each path
      Int res1_size = 1; // (res1_atom_indices.array() != -1).count();
      Int res2_size = 1; // (res2_atom_indices.array() != -1).count();

      // Try both unique and wildcard IDs for block 1
      for (bool wildcard : {false, true}) {
        // Get the lookup tables for atom ID
        const auto& res1_atom_id_table = (wildcard)
                                              ? atom_wildcard_ids[block_type1]
                                              : atom_unique_ids[block_type1];
        const auto& res2_atom_id_table = atom_wildcard_ids[block_type2];

        // Get the atom IDs
        Vec<Int, 3> res1_subgraph_atom_ids =
            get_atom_ids(res1_atom_id_table, res1_path);
        Vec<Int, 3> res2_subgraph_atom_ids =
            get_atom_ids(res2_atom_id_table, res2_path);

        // Make the joined data structures
        Vec<Int, 4> path;
        Vec<Int, 4> atom_indices;
        Vec<Int, 4> subgraph_atom_ids;

        // Init with -1s
        path << -1, -1, -1, -1;
        atom_indices << -1, -1, -1, -1;
        subgraph_atom_ids << -1, -1, -1, -1;

        // Join the paths into 1
        path.head(res1_size + res2_size) << res1_path.tail(res1_size),
            res2_path.head(res2_size);
        atom_indices.head(res1_size + res2_size)
            << res1_atom_indices.tail(res1_size),
            res2_atom_indices.head(res2_size);
        subgraph_atom_ids.head(res1_size + res2_size)
            << res1_subgraph_atom_ids.tail(res1_size),
            res2_subgraph_atom_ids.head(res2_size);

        // Do the lookup
        int param_index =
            hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
        if (param_index != -1) {
          // we found it!
          subgraph_atom_indices = atom_indices;
          break;
        }
      }
    }

    if (param_index != -1) {
      // score_subgraph(subgraph_atom_indices, param_index);
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[subgraph_atom_indices[0]];
      Vec<Real, 3> atom2 = rot_coords[subgraph_atom_indices[1]];

      int score_type = params[0];
      Real block_weight = dTdV[score_type][dispatch_ind];

      int V_ind = dispatch_ind;

      auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
      accumulate_result<Real, Int, 2, D>(
          eval,
          subgraph_atom_indices.head(2),
          V[score_type][V_ind],
          dV_dx[score_type],
          block_weight);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
    rotconn_for_lengths.size(0), eval_lengths);


  auto eval_angles = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rotconn_ind = rotconn_for_angles[index];
    int const rot_ind1 = rotconn_ind / (n_max_conns + 1);
    int const conn_ind1 = rotconn_ind % (n_max_conns + 1);
    int const rotconn_angle_offset = count_n_at_trip_angls_for_rotconn_offset[rotconn_ind];
    int const local_rot_and_angle_ind_for_rotconn = index - rotconn_angle_offset;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    int param_index = -1;
    Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

    int const rotconn_offset = n_intxns_for_rot_conn_offset[rotconn_ind];
    int dispatch_ind;

    if (conn_ind1 == n_max_conns) {
      // intra-residue!

      // There is only one interaction for this conn_ind1, so the
      // "dispatch index" is equal to the rotconn_offset.
      dispatch_ind = rotconn_offset;

      // Get the subgraph for this particular angle we are looking at
      int const subgraph_offset = cart_subgraph_offsets[block_type1] +
        cart_subgraph_type_offsets[block_type1][subgraph_angle];
      int const subgraph_ind = subgraph_offset + local_rot_and_angle_ind_for_rotconn;

      for (bool reverse : {false, true}) {
        Vec<Int, 4> subgraph = cart_subgraphs[subgraph_ind];
        if (reverse) reverse_subgraph(subgraph);

        Vec<Int, 4> subgraph_atom_ids =
            get_atom_ids(atom_unique_ids[block_type1], subgraph);
        param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

        subgraph_atom_indices = atom_local_to_global_indices(subgraph, rot_coord_offset1);

        if (param_index != -1){
          break;
        }
      }
    } else {
      // inter residue!
      int const block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][0];
      int const conn_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][1];

      // which rotamer on the other block?
      // local_length_ind_for_rotconn / n_angles
      int const n_intra_angles = get_n_connection_spanning_subgraphs(subgraph_angle);
      int const local_rot_ind2 = local_rot_and_angle_ind_for_rotconn / n_intra_angles;
      int const local_angle_ind = local_rot_and_angle_ind_for_rotconn % n_intra_angles;

      // The "dispatch index" is rotconn_offset + the local index of the
      // block2 rotamer
      dispatch_ind = rotconn_offset + local_rot_ind2;

      // which angle for this rotconn?

      int const subgraph_offset = get_connection_spanning_subgraphs_offset(subgraph_angle);
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // the index of this subgraph in the list of inter-block subgraphs
      // from connection.hh. We do not know if this angle is defined
      // for this pair of block types yet, so we will attempt to resolve
      // the atom indices, and score only if the resolution succeeds
      int const subgraph_index = subgraph_offset + local_angle_ind;

      tuple<int, int, int> spanning_subgraphs =
          get_connection_spanning_subgraph_indices(subgraph_index);
      int res1_path_ind = common::get<1>(spanning_subgraphs);
      int res2_path_ind = common::get<2>(spanning_subgraphs);

      // Grab the paths from each block
      Vec<Int, 3> res1_path =
          atom_paths_from_conn[block_type1][conn_ind1][res1_path_ind];
      Vec<Int, 3> res2_path =
          atom_paths_from_conn[block_type2][conn_ind2][res2_path_ind];

      // Now make sure these are valid paths
      if (res1_path[0] == -1 || res2_path[0] == -1) return;
      // Reverse the first path so that we can join them head-to-head
      res1_path.reverseInPlace();
      // Get a new Vec containing the global indices of the atoms
      Vec<Int, 3> res1_atom_indices =
          atom_local_to_global_indices(res1_path, rot_coord_offset1);
      Vec<Int, 3> res2_atom_indices =
          atom_local_to_global_indices(res2_path, rot_coord_offset2);
      // Calculate the size of each path
      Int res1_size = 1; // (res1_atom_indices.array() != -1).count();
      Int res2_size = 1; // (res2_atom_indices.array() != -1).count();

      // Try both unique and wildcard IDs for block 1
      for (bool wildcard : {false, true}) {
        // Get the lookup tables for atom ID
        const auto& res1_atom_id_table = (wildcard)
                                              ? atom_wildcard_ids[block_type1]
                                              : atom_unique_ids[block_type1];
        const auto& res2_atom_id_table = atom_wildcard_ids[block_type2];

        // Get the atom IDs
        Vec<Int, 3> res1_subgraph_atom_ids =
            get_atom_ids(res1_atom_id_table, res1_path);
        Vec<Int, 3> res2_subgraph_atom_ids =
            get_atom_ids(res2_atom_id_table, res2_path);

        // Make the joined data structures
        Vec<Int, 4> path;
        Vec<Int, 4> atom_indices;
        Vec<Int, 4> subgraph_atom_ids;

        // Init with -1s
        path << -1, -1, -1, -1;
        atom_indices << -1, -1, -1, -1;
        subgraph_atom_ids << -1, -1, -1, -1;

        // Join the paths into 1
        path.head(res1_size + res2_size) << res1_path.tail(res1_size),
            res2_path.head(res2_size);
        atom_indices.head(res1_size + res2_size)
            << res1_atom_indices.tail(res1_size),
            res2_atom_indices.head(res2_size);
        subgraph_atom_ids.head(res1_size + res2_size)
            << res1_subgraph_atom_ids.tail(res1_size),
            res2_subgraph_atom_ids.head(res2_size);

        // Do the lookup
        int param_index =
            hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
        if (param_index != -1) {
          // we found it!
          subgraph_atom_indices = atom_indices;
          break;
        }
      }
    }

    if (param_index != -1) {
      // score_subgraph(subgraph_atom_indices, param_index);
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[subgraph_atom_indices[0]];
      Vec<Real, 3> atom2 = rot_coords[subgraph_atom_indices[1]];
      Vec<Real, 3> atom3 = rot_coords[subgraph_atom_indices[2]];

      int score_type = params[0];
      Real block_weight = dTdV[score_type][dispatch_ind];

      int V_ind = dispatch_ind;

      auto eval =
          cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
      accumulate_result<Real, Int, 3, D>(
          eval,
          subgraph_atom_indices.head(3),
          V[score_type][V_ind],
          dV_dx[score_type],
          block_weight);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
    rotconn_for_angles.size(0), eval_angles);


  // And finally, torsions!
  auto eval_torsions = ([=] TMOL_DEVICE_FUNC (int index) {
    int const rotconn_ind = rotconn_for_torsions[index];
    int const rot_ind1 = rotconn_ind / (n_max_conns + 1);
    int const conn_ind1 = rotconn_ind % (n_max_conns + 1);
    int const rotconn_torsion_offset = count_n_at_quad_dihes_for_rotconn_offset[rotconn_ind];
    int const local_rot_and_torsion_ind_for_rotconn = index - rotconn_torsion_offset;

    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_ind1 = block_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    int param_index = -1;
    Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

    int const rotconn_offset = n_intxns_for_rot_conn_offset[rotconn_ind];
    int dispatch_ind;

    if (conn_ind1 == n_max_conns) {
      // intra-residue!

      // There is only one interaction for this conn_ind1, so the
      // "dispatch index" is equal to the rotconn_offset.
      dispatch_ind = rotconn_offset;

      // Get the subgraph for this particular angle we are looking at
      int const subgraph_offset = cart_subgraph_offsets[block_type1] +
        cart_subgraph_type_offsets[block_type1][subgraph_torsion];
      int const subgraph_ind = subgraph_offset + local_rot_and_torsion_ind_for_rotconn;

      for (bool reverse : {false, true}) {
        Vec<Int, 4> subgraph = cart_subgraphs[subgraph_ind];
        if (reverse) reverse_subgraph(subgraph);

        Vec<Int, 4> subgraph_atom_ids =
            get_atom_ids(atom_unique_ids[block_type1], subgraph);
        param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

        subgraph_atom_indices = atom_local_to_global_indices(subgraph, rot_coord_offset1);

        if (param_index != -1){
          break;
        }
      }
    } else {
      // inter residue!
      int const block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][0];
      int const conn_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1][1];

      // which rotamer on the other block?
      // which torsion for this rotconn?
      int const n_intra_torsions = get_n_connection_spanning_subgraphs(subgraph_torsion);
      int const local_rot_ind2 = local_rot_and_torsion_ind_for_rotconn / n_intra_torsions;
      int const local_torsion_ind = local_rot_and_torsion_ind_for_rotconn % n_intra_torsions;

      // The "dispatch index" is rotconn_offset + the local index of the
      // block2 rotamer
      dispatch_ind = rotconn_offset + local_rot_ind2;


      int const subgraph_offset = get_connection_spanning_subgraphs_offset(subgraph_torsion);
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // the index of this subgraph in the list of inter-block subgraphs
      // from connection.hh. We do not know if this angle is defined
      // for this pair of block types yet, so we will attempt to resolve
      // the atom indices, and score only if the resolution succeeds
      int const subgraph_index = subgraph_offset + local_torsion_ind;

      tuple<int, int, int> spanning_subgraphs =
          get_connection_spanning_subgraph_indices(subgraph_index);
      int res1_path_ind = common::get<1>(spanning_subgraphs);
      int res2_path_ind = common::get<2>(spanning_subgraphs);

      // Grab the paths from each block
      Vec<Int, 3> res1_path =
          atom_paths_from_conn[block_type1][conn_ind1][res1_path_ind];
      Vec<Int, 3> res2_path =
          atom_paths_from_conn[block_type2][conn_ind2][res2_path_ind];

      // Now make sure these are valid paths
      if (res1_path[0] == -1 || res2_path[0] == -1) return;
      // Reverse the first path so that we can join them head-to-head
      res1_path.reverseInPlace();
      // Get a new Vec containing the global indices of the atoms
      Vec<Int, 3> res1_atom_indices =
          atom_local_to_global_indices(res1_path, rot_coord_offset1);
      Vec<Int, 3> res2_atom_indices =
          atom_local_to_global_indices(res2_path, rot_coord_offset2);
      // Calculate the size of each path
      Int res1_size = 1; // (res1_atom_indices.array() != -1).count();
      Int res2_size = 1; // (res2_atom_indices.array() != -1).count();

      // Try both unique and wildcard IDs for block 1
      for (bool wildcard : {false, true}) {
        // Get the lookup tables for atom ID
        const auto& res1_atom_id_table = (wildcard)
                                              ? atom_wildcard_ids[block_type1]
                                              : atom_unique_ids[block_type1];
        const auto& res2_atom_id_table = atom_wildcard_ids[block_type2];

        // Get the atom IDs
        Vec<Int, 3> res1_subgraph_atom_ids =
            get_atom_ids(res1_atom_id_table, res1_path);
        Vec<Int, 3> res2_subgraph_atom_ids =
            get_atom_ids(res2_atom_id_table, res2_path);

        // Make the joined data structures
        Vec<Int, 4> path;
        Vec<Int, 4> atom_indices;
        Vec<Int, 4> subgraph_atom_ids;

        // Init with -1s
        path << -1, -1, -1, -1;
        atom_indices << -1, -1, -1, -1;
        subgraph_atom_ids << -1, -1, -1, -1;

        // Join the paths into 1
        path.head(res1_size + res2_size) << res1_path.tail(res1_size),
            res2_path.head(res2_size);
        atom_indices.head(res1_size + res2_size)
            << res1_atom_indices.tail(res1_size),
            res2_atom_indices.head(res2_size);
        subgraph_atom_ids.head(res1_size + res2_size)
            << res1_subgraph_atom_ids.tail(res1_size),
            res2_subgraph_atom_ids.head(res2_size);

        // Do the lookup
        int param_index =
            hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
        if (param_index != -1) {
          // we found it!
          subgraph_atom_indices = atom_indices;
          break;
        }
      }
    }

    if (param_index != -1) {
      // score_subgraph(subgraph_atom_indices, param_index);
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[subgraph_atom_indices[0]];
      Vec<Real, 3> atom2 = rot_coords[subgraph_atom_indices[1]];
      Vec<Real, 3> atom3 = rot_coords[subgraph_atom_indices[2]];
      Vec<Real, 3> atom4 = rot_coords[subgraph_atom_indices[3]];

      int score_type = params[0];
      Real block_weight = dTdV[score_type][dispatch_ind];

      int V_ind = dispatch_ind;

      auto eval = cbtorsion_V_dV(
          atom1,
          atom2,
          atom3,
          atom4,
          params[1],
          params[2],
          params[3],
          params[4],
          params[5],
          params[6]);
      accumulate_result<Real, Int, 4, D>(
          eval,
          subgraph_atom_indices.head(4),
          V[score_type][V_ind],
          dV_dx[score_type],
          block_weight);
    }

  });
  DeviceDispatch<D>::template forall<launch_t>(
    rotconn_for_torsions.size(0), eval_torsions);

  return dV_dx_t;





  // //////////////////////////// OLD CODE ////////////////////////////////
  // // auto V_t = TPack<Real, 2, D>::zeros({5, n_poses});
  // //  auto V_t = TPack<Real, 2, D>::zeros({1, n_poses});
  // TPack<Real, 4, D> V_t;
  // V_t = TPack<Real, 4, D>::zeros({5, n_poses, n_blocks, n_blocks});

  // auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({5, n_poses, n_max_atoms});

  // auto V = V_t.view;
  // auto dV_dx = dV_dx_t.view;

  // max_subgraphs_per_block +=
  //     NUM_INTER_RES_PATHS;  // Add in the inter-residue subgraphs
      
  // auto func = ([=] TMOL_DEVICE_FUNC(
  //                  int pose_index, int block_index, int subgraph_index) {
  //   Real score = 0;

  //   int block_type = pose_stack_block_type[pose_index][block_index];
  //   auto pose_coords = coords[pose_index];
  //   int block_coord_offset =
  //       pose_stack_block_coord_offset[pose_index][block_index];
  //   if (block_type < 0) {
  //     return;
  //   }
  //   int subgraph_offset = cart_subgraph_offsets[block_type];
  //   int subgraph_offset_next = block_type + 1 == n_block_types
  //                                  ? n_subgraphs
  //                                  : cart_subgraph_offsets[block_type + 1];
  //   subgraph_index += subgraph_offset;

  //   auto score_subgraph =
  //       ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
  //         Vec<Real, 7> params = hash_values[param_index];

  //         Vec<Real, 3> atom1 = pose_coords[atoms[0]];
  //         Vec<Real, 3> atom2 = pose_coords[atoms[1]];

  //         subgraph_type type = get_subgraph_type(atoms);

  //         int score_type = params[0];

  //         Real block_weight =
  //             (dTdV[score_type][pose_index][block_index][block_index]);

  //         // Real score;
  //         switch (type) {
  //           case subgraph_type::length: {
  //             auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
  //             accumulate_result<Real, Int, 2, D>(
  //                 eval,
  //                 atoms.head(2),
  //                 V[score_type][pose_index][block_index][block_index],
  //                 dV_dx[score_type][pose_index],
  //                 block_weight);

  //             break;
  //           }
  //           case subgraph_type::angle: {
  //             Vec<Real, 3> atom3 = pose_coords[atoms[2]];
  //             auto eval =
  //                 cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
  //             accumulate_result<Real, Int, 3, D>(
  //                 eval,
  //                 atoms.head(3),
  //                 V[score_type][pose_index][block_index][block_index],
  //                 dV_dx[score_type][pose_index],
  //                 block_weight);

  //             break;
  //           }
  //           case subgraph_type::torsion: {
  //             Vec<Real, 3> atom3 = pose_coords[atoms[2]];
  //             Vec<Real, 3> atom4 = pose_coords[atoms[3]];
  //             auto eval = cbtorsion_V_dV(
  //                 atom1,
  //                 atom2,
  //                 atom3,
  //                 atom4,
  //                 params[1],
  //                 params[2],
  //                 params[3],
  //                 params[4],
  //                 params[5],
  //                 params[6]);
  //             accumulate_result<Real, Int, 4, D>(
  //                 eval,
  //                 atoms.head(4),
  //                 V[score_type][pose_index][block_index][block_index],
  //                 dV_dx[score_type][pose_index],
  //                 block_weight);

  //             break;
  //           }
  //         }
  //       });

  //   if (subgraph_index >= subgraph_offset_next) {
  //     if (subgraph_index >= subgraph_offset_next + NUM_INTER_RES_PATHS) return;

  //     subgraph_index -= subgraph_offset_next;

  //     // Iterate over each connection in this block. We don't need to worry
  //     // about the other block accidentally duplicating the energy attribution
  //     // since the ordering of the atoms matters.
  //     for (int i = 0; i < n_max_conns; i++) {
  //       const Vec<Int, 2>& connection =
  //           pose_stack_inter_block_connections[pose_index][block_index][i];
  //       int other_block_index = connection[0];
  //       // No block on the other side of the connection, nothing to do
  //       if (other_block_index == -1) continue;
  //       int other_block_type =
  //           pose_stack_block_type[pose_index][other_block_index];
  //       int other_block_offset =
  //           pose_stack_block_coord_offset[pose_index][other_block_index];

  //       int other_connection_index = connection[1];

  //       // From our subgraph index, grab the corresponding paths indices for
  //       // each block
  //       tuple<int, int> spanning_subgraphs =
  //           get_connection_spanning_subgraph_indices(subgraph_index);
  //       int res1_path_ind = common::get<0>(spanning_subgraphs);
  //       int res2_path_ind = common::get<1>(spanning_subgraphs);

  //       // Grab the paths from each block
  //       Vec<Int, 3> res1_path =
  //           atom_paths_from_conn[block_type][i][res1_path_ind];
  //       Vec<Int, 3> res2_path =
  //           atom_paths_from_conn[other_block_type][other_connection_index]
  //                               [res2_path_ind];

  //       // Make sure these are valid paths
  //       if (res1_path[0] == -1 || res2_path[0] == -1) continue;
  //       // Reverse the first path so that we can join them head-to-head
  //       res1_path.reverseInPlace();

  //       // Get a new Vec containing the global indices of the atoms
  //       Vec<Int, 3> res1_atom_indices =
  //           atom_local_to_global_indices(res1_path, block_coord_offset);
  //       Vec<Int, 3> res2_atom_indices =
  //           atom_local_to_global_indices(res2_path, other_block_offset);

  //       // Calculate the size of each path
  //       Int res1_size = (res1_atom_indices.array() != -1).count();
  //       Int res2_size = (res2_atom_indices.array() != -1).count();

  //       // Try both unique and wildcard IDs for block 1
  //       for (bool wildcard : {false, true}) {
  //         // Get the lookup tables for atom ID
  //         const auto& res1_atom_id_table = (wildcard)
  //                                              ? atom_wildcard_ids[block_type]
  //                                              : atom_unique_ids[block_type];
  //         const auto& res2_atom_id_table = atom_wildcard_ids[other_block_type];

  //         // Get the atom IDs
  //         Vec<Int, 3> res1_subgraph_atom_ids =
  //             get_atom_ids(res1_atom_id_table, res1_path);
  //         Vec<Int, 3> res2_subgraph_atom_ids =
  //             get_atom_ids(res2_atom_id_table, res2_path);

  //         // Make the joined data structures
  //         Vec<Int, 4> path;
  //         Vec<Int, 4> atom_indices;
  //         Vec<Int, 4> subgraph_atom_ids;

  //         // Init with -1s
  //         path << -1, -1, -1, -1;
  //         atom_indices << -1, -1, -1, -1;
  //         subgraph_atom_ids << -1, -1, -1, -1;

  //         // Join the paths into 1
  //         path.head(res1_size + res2_size) << res1_path.tail(res1_size),
  //             res2_path.head(res2_size);
  //         atom_indices.head(res1_size + res2_size)
  //             << res1_atom_indices.tail(res1_size),
  //             res2_atom_indices.head(res2_size);
  //         subgraph_atom_ids.head(res1_size + res2_size)
  //             << res1_subgraph_atom_ids.tail(res1_size),
  //             res2_subgraph_atom_ids.head(res2_size);

  //         // Do the lookup
  //         int param_index =
  //             hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

  //         // If we found a param that matches, score the subgraph
  //         if (param_index != -1) {
  //           score_subgraph(atom_indices, param_index);
  //         }
  //       }
  //     }
  //     return;
  //   }

  //   // Intra-res subgraphs
  //   for (bool reverse : {false, true}) {
  //     Vec<Int, 4> subgraph = cart_subgraphs[subgraph_index];
  //     if (reverse) reverse_subgraph(subgraph);

  //     Vec<Int, 4> subgraph_atom_ids =
  //         get_atom_ids(atom_unique_ids[block_type], subgraph);
  //     int param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

  //     Vec<Int, 4> subgraph_atom_indices =
  //         atom_local_to_global_indices(subgraph, block_coord_offset);

  //     if (param_index != -1) {
  //       score_subgraph(subgraph_atom_indices, param_index);
  //     }
  //   }
  // });

  // DeviceDispatch<D>::foreach_combination_triple(
  //     n_poses, n_blocks, max_subgraphs_per_block, func);

  // return dV_dx_t;

}  // namespace potentials

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
