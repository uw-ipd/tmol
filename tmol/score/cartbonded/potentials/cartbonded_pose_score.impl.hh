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
    Real& val,
    common::tuple<Real, Vec<Real3, N>> to_add,
    Vec<Int, N> atoms,
    bool accumulate_derivs,
    TensorAccessor<Vec<Real, 3>, 1, D> dV,
    const Real& weight = 1.0) {
  val += common::get<0>(to_add);
  if (accumulate_derivs) {
    for (int i = 0; i < N; i++) {
      accumulate<D, Vec<Real, 3>>::add(
          dV[atoms[i]], common::get<1>(to_add)[i] * weight);
    }
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
        TPack<Real, 4, D>,  // V_t: n-terms x n_poses x [n_blocks x n_blocks or
                            // 1 x 1]
        TPack<Vec<Real, 3>, 2, D>  // dV_dx_t,
        > {
  using tmol::score::common::get_connection_spanning_subgraphs_offset;
  using tmol::score::common::get_n_connection_spanning_subgraphs;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
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
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
  assert(atom_paths_from_conn.size(2) == MAX_PATHS_FROM_CONN);

  assert(atom_unique_ids.size(0) == n_block_types);
  assert(atom_unique_ids.size(1) == n_max_atoms_per_block);

  assert(atom_wildcard_ids.size(0) == n_block_types);
  assert(atom_wildcard_ids.size(1) == n_max_atoms_per_block);

  assert(cart_subgraph_offsets.size(0) == n_block_types);
  assert(cart_subgraph_type_counts.size(0) == n_block_types);
  assert(cart_subgraph_type_offsets.size(0) == n_block_types);

  // Algorithm: launch n_rots * (max_n_conns + 1) CTAs, each looking
  // at a single residue's (rotamer's) connection (+ one "connection" to self),
  // killing CTAs that connect to lower residues to avoid double counting.
  // Each CTA iterates across all the subgraphs for that connection,
  // then performs a reduction so that thread-0 can write the results to global
  // memory.

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // Convention: conn == max_n_conns ==> the intra-rotamer set of cart energies

  // Allocate the tensors to which we will write our outputs
  int const n_V = output_block_pair_energies ? max_n_blocks : 1;
  auto V_t = TPack<Real, 4, D>::zeros({5, n_poses, n_V, n_V});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    // Only one element of this union: the shared memory array for
    // the reduction, which itself only will take any space if NT > 32
    // as the reduction otherwise uses only warp shuffle operations!
    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      int stub;  // can't have an empty union, so, we'll have an integer
                 // placeholder.
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    int const pose_ind = cta / (max_n_blocks * (max_n_conns + 1));
    int const block_conn = cta % (max_n_blocks * (max_n_conns + 1));
    int const block_ind1 = block_conn / (max_n_conns + 1);
    int const conn_ind1 = block_conn % (max_n_conns + 1);
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];

    if (block_type1 == -1) {
      return;
    }

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real length_score = 0.0;
    Real angle_score = 0.0;
    Real torsion_score = 0.0;
    Real improper_torsion_score = 0.0;
    Real hxyl_torsion_score = 0.0;

    auto score_subgraph =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
          Vec<Real, 7> params = hash_values[param_index];

          Vec<Real, 3> atom1 = rot_coords[atoms[0]];
          Vec<Real, 3> atom2 = rot_coords[atoms[1]];

          subgraph_type type = get_subgraph_type(atoms);

          int score_type = params[0];

          // Real score;
          switch (type) {
            case subgraph_type::length: {
              auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
              accumulate_result<Real, Int, 2, D>(
                  length_score,
                  eval,
                  atoms.head(2),
                  compute_derivs,
                  dV_dx[score_type],
                  1.0);

              break;
            }
            case subgraph_type::angle: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              auto eval =
                  cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
              accumulate_result<Real, Int, 3, D>(
                  angle_score,
                  eval,
                  atoms.head(3),
                  compute_derivs,
                  dV_dx[score_type],
                  1.0);

              break;
            }
            case subgraph_type::torsion: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              Vec<Real, 3> atom4 = rot_coords[atoms[3]];
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
              Real& tor_score = (score_type == 3)   ? improper_torsion_score
                                : (score_type == 4) ? hxyl_torsion_score
                                                    : torsion_score;
              accumulate_result<Real, Int, 4, D>(
                  tor_score,
                  eval,
                  atoms.head(4),
                  compute_derivs,
                  dV_dx[score_type],
                  1.0);

              break;
            }
          }
        });

    int block_ind2 = -1;
    if (conn_ind1 == max_n_conns) {
      // intra-residue!
      block_ind2 = block_ind1;
      // Threads in the CTA iterate across the subgraphs within the block
      int subgraph_offset = cart_subgraph_offsets[block_type1];
      int subgraph_offset_next = block_type1 + 1 == n_block_types
                                     ? n_subgraphs
                                     : cart_subgraph_offsets[block_type1 + 1];
      int n_subgraphs = subgraph_offset_next - subgraph_offset;
      auto eval_intra_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        // printf(" Intra res interaction for rot %d (bt %d); cta %d tid %d \n",
        // rot_ind1, block_type1, cta, tid);

        for (int i = tid; i < n_subgraphs; i += nt) {
          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};
          for (bool reverse : {false, true}) {
            Vec<Int, 4> subgraph = cart_subgraphs[subgraph_offset + i];
            if (reverse) reverse_subgraph(subgraph);

            Vec<Int, 4> subgraph_atom_ids =
                get_atom_ids(atom_unique_ids[block_type1], subgraph);
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

            subgraph_atom_indices =
                atom_local_to_global_indices(subgraph, rot_coord_offset1);

            if (param_index != -1) {
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_intra_res_subgraphs);
    } else {
      // Inter residue!
      block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1]
                                                     [conn_ind1][0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      if (block_ind2 == -1) {
        return;
      }
      if (block_ind1 > block_ind2) {
        // to avoid double counting, only have the lower-indexed block
        // handle the interaction
        return;
      }
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // Use capture-by-reference here so that we can write to the length_score,
      // angle_score, and torsion_score variables
      auto eval_inter_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        int n_connection_spanning_subgraphs = common::NUM_INTER_RES_PATHS;
        for (int i = tid; i < 2 * n_connection_spanning_subgraphs; i += nt) {
          bool reverse = i % 2 == 1;
          // interleave the subgraphs from the two directions so we can have
          // better warp coherence.
          int const subgraph_index = i / 2;
          int const block_typeA = reverse ? block_type2 : block_type1;
          int const block_typeB = reverse ? block_type1 : block_type2;
          int const rot_coord_offsetA =
              reverse ? rot_coord_offset2 : rot_coord_offset1;
          int const rot_coord_offsetB =
              reverse ? rot_coord_offset1 : rot_coord_offset2;
          int const conn_indA = reverse ? conn_ind2 : conn_ind1;
          int const conn_indB = reverse ? conn_ind1 : conn_ind2;

          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

          tuple<int, int, int> spanning_subgraphs =
              get_connection_spanning_subgraph_indices(subgraph_index);
          int resA_path_ind = common::get<1>(spanning_subgraphs);
          int resB_path_ind = common::get<2>(spanning_subgraphs);
          // Grab the paths from each block
          Vec<Int, 3> resA_path =
              atom_paths_from_conn[block_typeA][conn_indA][resA_path_ind];
          Vec<Int, 3> resB_path =
              atom_paths_from_conn[block_typeB][conn_indB][resB_path_ind];

          // Make sure these are valid paths
          if (resA_path[0] == -1 || resB_path[0] == -1) continue;
          // Reverse the first path so that we can join them head-to-head
          resA_path.reverseInPlace();
          // Get a new Vec containing the global indices of the atoms
          Vec<Int, 3> resA_atom_indices =
              atom_local_to_global_indices(resA_path, rot_coord_offsetA);
          Vec<Int, 3> resB_atom_indices =
              atom_local_to_global_indices(resB_path, rot_coord_offsetB);
          // Calculate the size of each path
          Int resA_size = (resA_atom_indices.array() != -1).count();
          Int resB_size = (resB_atom_indices.array() != -1).count();

          // Try both unique and wildcard IDs for block A
          for (bool wildcard : {false, true}) {
            // Get the lookup tables for atom ID
            const auto& resA_atom_id_table =
                (wildcard) ? atom_wildcard_ids[block_typeA]
                           : atom_unique_ids[block_typeA];
            const auto& resB_atom_id_table = atom_wildcard_ids[block_typeB];

            // Get the atom IDs
            Vec<Int, 3> resA_subgraph_atom_ids =
                get_atom_ids(resA_atom_id_table, resA_path);
            Vec<Int, 3> resB_subgraph_atom_ids =
                get_atom_ids(resB_atom_id_table, resB_path);

            // Make the joined data structures
            Vec<Int, 4> path;
            Vec<Int, 4> atom_indices;
            Vec<Int, 4> subgraph_atom_ids;

            // Init with -1s
            path << -1, -1, -1, -1;
            atom_indices << -1, -1, -1, -1;
            subgraph_atom_ids << -1, -1, -1, -1;

            // Join the paths into 1
            path.head(resA_size + resB_size) << resA_path.tail(resA_size),
                resB_path.head(resB_size);
            atom_indices.head(resA_size + resB_size)
                << resA_atom_indices.tail(resA_size),
                resB_atom_indices.head(resB_size);
            subgraph_atom_ids.head(resA_size + resB_size)
                << resA_subgraph_atom_ids.tail(resA_size),
                resB_subgraph_atom_ids.head(resB_size);

            // Do the lookup
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
            if (param_index != -1) {
              // we found it!
              subgraph_atom_indices = atom_indices;
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_subgraphs);
    }

    // Now let's accumulate the scores into main memory
    auto reduce_energies = ([&] TMOL_DEVICE_FUNC(int tid) {
      Real const cta_length_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              length_score, shared, mgpu::plus_t<Real>());
      Real const cta_angle_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              angle_score, shared, mgpu::plus_t<Real>());
      Real const cta_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              torsion_score, shared, mgpu::plus_t<Real>());
      Real const cta_improper_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              improper_torsion_score, shared, mgpu::plus_t<Real>());
      Real const cta_hxyl_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              hxyl_torsion_score, shared, mgpu::plus_t<Real>());

      if (tid == 0) {
        // straight assignment in the block-pair scoring case; atomic add
        // otherwise
        if (output_block_pair_energies) {
          if (cta_length_score != 0.0) {
            V[0][pose_ind][block_ind1][block_ind2] = cta_length_score;
          }
          if (cta_angle_score != 0.0) {
            V[1][pose_ind][block_ind1][block_ind2] = cta_angle_score;
          }
          if (cta_torsion_score != 0.0) {
            V[2][pose_ind][block_ind1][block_ind2] = cta_torsion_score;
          }
          if (cta_improper_torsion_score != 0.0) {
            V[3][pose_ind][block_ind1][block_ind2] = cta_improper_torsion_score;
          }
          if (cta_hxyl_torsion_score != 0.0) {
            V[4][pose_ind][block_ind1][block_ind2] = cta_hxyl_torsion_score;
          }
        } else {
          if (cta_length_score != 0.0) {
            accumulate<D, Real>::add(V[0][pose_ind][0][0], cta_length_score);
          }
          if (cta_angle_score != 0.0) {
            accumulate<D, Real>::add(V[1][pose_ind][0][0], cta_angle_score);
          }
          if (cta_torsion_score != 0.0) {
            accumulate<D, Real>::add(V[2][pose_ind][0][0], cta_torsion_score);
          }
          if (cta_improper_torsion_score != 0.0) {
            accumulate<D, Real>::add(
                V[3][pose_ind][0][0], cta_improper_torsion_score);
          }
          if (cta_hxyl_torsion_score != 0.0) {
            accumulate<D, Real>::add(
                V[4][pose_ind][0][0], cta_hxyl_torsion_score);
          }
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
  });
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      n_poses * max_n_blocks * (max_n_conns + 1),
      eval_subgraphs_for_interaction);

  return {V_t, dV_dx_t};
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

    // How many intra-block subgraphs of the three types (lengths, angles, &
    // torsions) are there?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_counts,
    // What are the _local_ offsets for each of the three types; i.e.
    // relative to the offset listed in cart_subgraph_offsets, where
    // do the subgraphs for each of the three types begin?
    TView<Vec<Int, 3>, 1, D> cart_subgraph_type_offsets,

    TView<Real, 4, D> dTdV  // nterms x n-poses x max_n_blocks x max_n_blocks
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
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
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
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
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms});
  auto dV_dx = dV_dx_t.view;

  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const pose_ind = cta / (max_n_blocks * (max_n_conns + 1));
    int const block_conn = cta % (max_n_blocks * (max_n_conns + 1));
    int const block_ind1 = block_conn / (max_n_conns + 1);
    int const conn_ind1 = block_conn % (max_n_conns + 1);
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];

    if (block_type1 == -1) {
      return;
    }

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    auto score_subgraph = ([=] TMOL_DEVICE_FUNC(
                               Vec<Int, 4> atoms,
                               Int param_index,
                               int pose_ind,
                               int block_ind1,
                               int block_ind2) {
      Real length_score = 0.0;
      Real angle_score = 0.0;
      Real torsion_score = 0.0;
      Real improper_torsion_score = 0.0;
      Real hxyl_torsion_score = 0.0;
      Vec<Real, 7> params = hash_values[param_index];

      Vec<Real, 3> atom1 = rot_coords[atoms[0]];
      Vec<Real, 3> atom2 = rot_coords[atoms[1]];

      subgraph_type type = get_subgraph_type(atoms);

      int score_type = params[0];
      Real block_weight = dTdV[score_type][pose_ind][block_ind1][block_ind2];

      // Real score;
      switch (type) {
        case subgraph_type::length: {
          auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
          accumulate_result<Real, Int, 2, D>(
              length_score,
              eval,
              atoms.head(2),
              true,
              dV_dx[score_type],
              block_weight);

          break;
        }
        case subgraph_type::angle: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          auto eval = cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
          accumulate_result<Real, Int, 3, D>(
              angle_score,
              eval,
              atoms.head(3),
              true,
              dV_dx[score_type],
              block_weight);

          break;
        }
        case subgraph_type::torsion: {
          Vec<Real, 3> atom3 = rot_coords[atoms[2]];
          Vec<Real, 3> atom4 = rot_coords[atoms[3]];
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
          Real& tor_score = (score_type == 3)   ? improper_torsion_score
                            : (score_type == 4) ? hxyl_torsion_score
                                                : torsion_score;
          accumulate_result<Real, Int, 4, D>(
              tor_score,
              eval,
              atoms.head(4),
              true,
              dV_dx[score_type],
              block_weight);

          break;
        }
      }
    });

    if (conn_ind1 == max_n_conns) {
      // intra-residue!
      int subgraph_offset = cart_subgraph_offsets[block_type1];
      int subgraph_offset_next = block_type1 + 1 == n_block_types
                                     ? n_subgraphs
                                     : cart_subgraph_offsets[block_type1 + 1];
      int n_subgraphs = subgraph_offset_next - subgraph_offset;
      auto eval_intra_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_subgraphs; i += nt) {
          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};
          for (bool reverse : {false, true}) {
            Vec<Int, 4> subgraph = cart_subgraphs[subgraph_offset + i];
            if (reverse) reverse_subgraph(subgraph);

            Vec<Int, 4> subgraph_atom_ids =
                get_atom_ids(atom_unique_ids[block_type1], subgraph);
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

            subgraph_atom_indices =
                atom_local_to_global_indices(subgraph, rot_coord_offset1);

            if (param_index != -1) {
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(
                subgraph_atom_indices,
                param_index,
                pose_ind,
                block_ind1,
                block_ind1);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_intra_res_subgraphs);
    } else {
      // Inter residue!
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      if (block_ind2 == -1) {
        return;
      }
      if (block_ind1 > block_ind2) {
        // to avoid double counting, only have the lower-indexed block
        // handle the interaction
        return;
      }
      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // Use by-reference capture here so that we can write to the length,
      // angle, and torsion_score variables
      auto eval_inter_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        int n_connection_spanning_subgraphs = common::NUM_INTER_RES_PATHS;
        for (int i = tid; i < 2 * n_connection_spanning_subgraphs; i += nt) {
          bool reverse = i % 2 == 1;
          // interleave the subgraphs from the two directions so we can have
          // better warp coherence.
          int const subgraph_index = i / 2;
          int const block_typeA = reverse ? block_type2 : block_type1;
          int const block_typeB = reverse ? block_type1 : block_type2;
          int const rot_coord_offsetA =
              reverse ? rot_coord_offset2 : rot_coord_offset1;
          int const rot_coord_offsetB =
              reverse ? rot_coord_offset1 : rot_coord_offset2;
          int const conn_indA = reverse ? conn_ind2 : conn_ind1;
          int const conn_indB = reverse ? conn_ind1 : conn_ind2;

          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

          tuple<int, int, int> spanning_subgraphs =
              get_connection_spanning_subgraph_indices(subgraph_index);
          int resA_path_ind = common::get<1>(spanning_subgraphs);
          int resB_path_ind = common::get<2>(spanning_subgraphs);
          // Grab the paths from each block
          Vec<Int, 3> resA_path =
              atom_paths_from_conn[block_typeA][conn_indA][resA_path_ind];
          Vec<Int, 3> resB_path =
              atom_paths_from_conn[block_typeB][conn_indB][resB_path_ind];

          // Make sure these are valid paths
          if (resA_path[0] == -1 || resB_path[0] == -1) continue;
          // Reverse the first path so that we can join them head-to-head
          resA_path.reverseInPlace();
          // Get a new Vec containing the global indices of the atoms
          Vec<Int, 3> resA_atom_indices =
              atom_local_to_global_indices(resA_path, rot_coord_offsetA);
          Vec<Int, 3> resB_atom_indices =
              atom_local_to_global_indices(resB_path, rot_coord_offsetB);
          // Calculate the size of each path
          Int resA_size = (resA_atom_indices.array() != -1).count();
          Int resB_size = (resB_atom_indices.array() != -1).count();

          // Try both unique and wildcard IDs for block A
          for (bool wildcard : {false, true}) {
            // Get the lookup tables for atom ID
            const auto& resA_atom_id_table =
                (wildcard) ? atom_wildcard_ids[block_typeA]
                           : atom_unique_ids[block_typeA];
            const auto& resB_atom_id_table = atom_wildcard_ids[block_typeB];

            // Get the atom IDs
            Vec<Int, 3> resA_subgraph_atom_ids =
                get_atom_ids(resA_atom_id_table, resA_path);
            Vec<Int, 3> resB_subgraph_atom_ids =
                get_atom_ids(resB_atom_id_table, resB_path);

            // Make the joined data structures
            Vec<Int, 4> path;
            Vec<Int, 4> atom_indices;
            Vec<Int, 4> subgraph_atom_ids;

            // Init with -1s
            path << -1, -1, -1, -1;
            atom_indices << -1, -1, -1, -1;
            subgraph_atom_ids << -1, -1, -1, -1;

            // Join the paths into 1
            path.head(resA_size + resB_size) << resA_path.tail(resA_size),
                resB_path.head(resB_size);
            atom_indices.head(resA_size + resB_size)
                << resA_atom_indices.tail(resA_size),
                resB_atom_indices.head(resB_size);
            subgraph_atom_ids.head(resA_size + resB_size)
                << resA_subgraph_atom_ids.tail(resA_size),
                resB_subgraph_atom_ids.head(resB_size);

            // Do the lookup
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
            if (param_index != -1) {
              // we found it!
              subgraph_atom_indices = atom_indices;
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(
                subgraph_atom_indices,
                param_index,
                pose_ind,
                block_ind1,
                block_ind2);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_subgraphs);
    }

    // We can skip the step where we accumulate into global memory because
    // we are only computing derivatives in this pass and don't need the score
    // a second time
  });
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      n_poses * max_n_blocks * (max_n_conns + 1),
      eval_subgraphs_for_interaction);

  return dV_dx_t;

}  // backward

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::forward(
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

    bool compute_derivs)
    -> std::tuple<
        TPack<Real, 2, D>,          // V_t,
        TPack<Vec<Real, 3>, 2, D>,  // dV_dx_t,
        TPack<Int, 2, D>,           // dispatch_indices_t,
        TPack<Int, 1, D>,           // n_output_intxns_for_rot_conn_offset,
        TPack<Int, 1, D>            // rotconn_for_output_intxn,
        > {
  using tmol::score::common::get_connection_spanning_subgraphs_offset;
  using tmol::score::common::get_n_connection_spanning_subgraphs;

  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
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
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
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
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  // Convention: conn == max_n_conns ==> the intra-rotamer set of cart energies
  int const max_n_interactions = n_rots * (max_n_conns + 1);

  // Here we only count the rotconns to "upper" neighbors
  auto n_output_intxns_for_rot_conn_t =
      TPack<Int, 1, D>::zeros({max_n_interactions});
  auto n_output_intxns_for_rot_conn = n_output_intxns_for_rot_conn_t.view;
  auto n_output_intxns_for_rot_conn_offset_t =
      TPack<Int, 1, D>::zeros({max_n_interactions});
  auto n_output_intxns_for_rot_conn_offset =
      n_output_intxns_for_rot_conn_offset_t.view;

  auto count_intxns_for_rot_conn = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rot_ind = index / (max_n_conns + 1);
    int const conn_ind = index % (max_n_conns + 1);

    int const pose_ind = pose_ind_for_rot[rot_ind];
    int const block_ind = block_ind_for_rot[rot_ind];
    int const block_type_ind = block_type_ind_for_rot[rot_ind];
    if (block_type_ind == -1) {
      // Not a real residue!
      return;
    }

    bool const is_intra_conn = conn_ind == max_n_conns;

    if (is_intra_conn) {
      n_output_intxns_for_rot_conn[index] = 1;
    } else {
      int const other_block_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn_ind][0];
      int const other_conn_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn_ind][1];

      if (other_block_ind == -1) {
        return;
      }
      int const other_block_n_rots =
          n_rots_for_block[pose_ind][other_block_ind];
      if (block_ind < other_block_ind) {
        n_output_intxns_for_rot_conn[index] = other_block_n_rots;
      }
    }
  });

  DeviceDispatch<D>::template forall<launch_t>(
      max_n_interactions, count_intxns_for_rot_conn);

  // Scan and LBS on n output intxns: figure out how many rotamer pair
  // interactions there are and which pair each work unit should be assigned to.
  int n_output_intxns_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_output_intxns_for_rot_conn.data(),
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions,
          mgpu::plus_t<Int>());
  TPack<Int, 1, D> rotconn_for_output_intxn_t =
      DeviceDispatch<D>::template load_balancing_search<launch_t>(
          n_output_intxns_total,
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions);
  auto rotconn_for_output_intxn = rotconn_for_output_intxn_t.view;

  // Allocate the tensors to which we will write our outputs
  int const n_V = output_block_pair_energies ? n_output_intxns_total : n_poses;
  auto V_t = TPack<Real, 2, D>::zeros({5, n_V});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms});
  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_output_intxns_total});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

  auto record_dispatch_indices_for_output_intxns =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const rotconn_ind = rotconn_for_output_intxn[index];
        int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
        int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
        int const pose_ind = pose_ind_for_rot[rot_ind1];
        dispatch_indices[0][index] = pose_ind;
        dispatch_indices[1][index] = rot_ind1;

        int rot_ind2;
        if (conn_ind1 == max_n_conns) {
          // intra-residue:
          rot_ind2 = rot_ind1;
        } else {
          // inter-residue
          int const block_ind1 = block_ind_for_rot[rot_ind1];
          int const rotconn_offset =
              n_output_intxns_for_rot_conn_offset[rotconn_ind];
          int const local_rot_ind2 = index - rotconn_offset;
          int const block_ind2 =
              pose_stack_inter_block_connections[pose_ind][block_ind1]
                                                [conn_ind1][0];

          rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
        }
        dispatch_indices[2][index] = rot_ind2;
      });
  // Record the rotamer pair indices for the output interactions; though we will
  // only use them downstream if we are in output_block_pair_energies mode
  // std::cout << "record_dispatch_indices_for_intxns" << std::endl;
  DeviceDispatch<D>::template forall<launch_t>(
      n_output_intxns_total, record_dispatch_indices_for_output_intxns);

  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    // Only one element of this union: the shared memory array for
    // the reduction, which itself only will take any space if NT > 32
    // as the reduction otherwise uses only warp shuffle operations!
    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      int stub;  // can't have an empty union, so, we'll have an integer
                 // placeholder.
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    int const rotconn_ind = rotconn_for_output_intxn[cta];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real length_score = 0.0;
    Real angle_score = 0.0;
    Real torsion_score = 0.0;
    Real improper_torsion_score = 0.0;
    Real hxyl_torsion_score = 0.0;

    auto score_subgraph =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
          Vec<Real, 7> params = hash_values[param_index];

          Vec<Real, 3> atom1 = rot_coords[atoms[0]];
          Vec<Real, 3> atom2 = rot_coords[atoms[1]];

          subgraph_type type = get_subgraph_type(atoms);

          int score_type = params[0];

          // Real score;
          switch (type) {
            case subgraph_type::length: {
              auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
              accumulate_result<Real, Int, 2, D>(
                  length_score,
                  eval,
                  atoms.head(2),
                  compute_derivs,
                  dV_dx[score_type],
                  1.0);

              break;
            }
            case subgraph_type::angle: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              auto eval =
                  cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
              accumulate_result<Real, Int, 3, D>(
                  angle_score,
                  eval,
                  atoms.head(3),
                  compute_derivs,
                  dV_dx[score_type],
                  1.0);

              break;
            }
            case subgraph_type::torsion: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              Vec<Real, 3> atom4 = rot_coords[atoms[3]];
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
              Real& tor_score = (score_type == 3)   ? improper_torsion_score
                                : (score_type == 4) ? hxyl_torsion_score
                                                    : torsion_score;
              accumulate_result<Real, Int, 4, D>(
                  tor_score,
                  eval,
                  atoms.head(4),
                  compute_derivs,
                  dV_dx[score_type],
                  1.0);

              break;
            }
          }
        });

    if (conn_ind1 == max_n_conns) {
      // intra-residue!
      int subgraph_offset = cart_subgraph_offsets[block_type1];
      int subgraph_offset_next = block_type1 + 1 == n_block_types
                                     ? n_subgraphs
                                     : cart_subgraph_offsets[block_type1 + 1];
      int n_subgraphs = subgraph_offset_next - subgraph_offset;
      auto eval_intra_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_subgraphs; i += nt) {
          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};
          for (bool reverse : {false, true}) {
            Vec<Int, 4> subgraph = cart_subgraphs[subgraph_offset + i];
            if (reverse) reverse_subgraph(subgraph);

            Vec<Int, 4> subgraph_atom_ids =
                get_atom_ids(atom_unique_ids[block_type1], subgraph);
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

            subgraph_atom_indices =
                atom_local_to_global_indices(subgraph, rot_coord_offset1);

            if (param_index != -1) {
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_intra_res_subgraphs);
    } else {
      // Inter residue!
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      int const local_rot_ind1 =
          rot_ind1 - rot_offset_for_block[pose_ind][block_ind1];
      int const local_rot_ind2 =
          cta - n_output_intxns_for_rot_conn_offset[rotconn_ind];
      int const rot_ind2 =
          rot_offset_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // Use by-reference capture here so that we can write to the length,
      // angle, and torsion_score variables
      auto eval_inter_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        int n_connection_spanning_subgraphs = common::NUM_INTER_RES_PATHS;
        for (int i = tid; i < 2 * n_connection_spanning_subgraphs; i += nt) {
          bool reverse = i % 2 == 1;
          // interleave the subgraphs from the two directions so we can have
          // better warp coherence.
          int const subgraph_index = i / 2;
          int const block_typeA = reverse ? block_type2 : block_type1;
          int const block_typeB = reverse ? block_type1 : block_type2;
          int const rot_coord_offsetA =
              reverse ? rot_coord_offset2 : rot_coord_offset1;
          int const rot_coord_offsetB =
              reverse ? rot_coord_offset1 : rot_coord_offset2;
          int const conn_indA = reverse ? conn_ind2 : conn_ind1;
          int const conn_indB = reverse ? conn_ind1 : conn_ind2;

          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

          tuple<int, int, int> spanning_subgraphs =
              get_connection_spanning_subgraph_indices(subgraph_index);
          int resA_path_ind = common::get<1>(spanning_subgraphs);
          int resB_path_ind = common::get<2>(spanning_subgraphs);
          // Grab the paths from each block
          Vec<Int, 3> resA_path =
              atom_paths_from_conn[block_typeA][conn_indA][resA_path_ind];
          Vec<Int, 3> resB_path =
              atom_paths_from_conn[block_typeB][conn_indB][resB_path_ind];

          // Make sure these are valid paths
          if (resA_path[0] == -1 || resB_path[0] == -1) continue;
          // Reverse the first path so that we can join them head-to-head
          resA_path.reverseInPlace();
          // Get a new Vec containing the global indices of the atoms
          Vec<Int, 3> resA_atom_indices =
              atom_local_to_global_indices(resA_path, rot_coord_offsetA);
          Vec<Int, 3> resB_atom_indices =
              atom_local_to_global_indices(resB_path, rot_coord_offsetB);
          // Calculate the size of each path
          Int resA_size = (resA_atom_indices.array() != -1).count();
          Int resB_size = (resB_atom_indices.array() != -1).count();

          // Try both unique and wildcard IDs for block A
          for (bool wildcard : {false, true}) {
            // Get the lookup tables for atom ID
            const auto& resA_atom_id_table =
                (wildcard) ? atom_wildcard_ids[block_typeA]
                           : atom_unique_ids[block_typeA];
            const auto& resB_atom_id_table = atom_wildcard_ids[block_typeB];

            // Get the atom IDs
            Vec<Int, 3> resA_subgraph_atom_ids =
                get_atom_ids(resA_atom_id_table, resA_path);
            Vec<Int, 3> resB_subgraph_atom_ids =
                get_atom_ids(resB_atom_id_table, resB_path);

            // Make the joined data structures
            Vec<Int, 4> path;
            Vec<Int, 4> atom_indices;
            Vec<Int, 4> subgraph_atom_ids;

            // Init with -1s
            path << -1, -1, -1, -1;
            atom_indices << -1, -1, -1, -1;
            subgraph_atom_ids << -1, -1, -1, -1;

            // Join the paths into 1
            path.head(resA_size + resB_size) << resA_path.tail(resA_size),
                resB_path.head(resB_size);
            atom_indices.head(resA_size + resB_size)
                << resA_atom_indices.tail(resA_size),
                resB_atom_indices.head(resB_size);
            subgraph_atom_ids.head(resA_size + resB_size)
                << resA_subgraph_atom_ids.tail(resA_size),
                resB_subgraph_atom_ids.head(resB_size);

            // Do the lookup
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
            if (param_index != -1) {
              // we found it!
              subgraph_atom_indices = atom_indices;
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_subgraphs);
    }

    // Now let's accumulate the scores into main memory
    auto reduce_energies = ([&] TMOL_DEVICE_FUNC(int tid) {
      Real const cta_length_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              length_score, shared, mgpu::plus_t<Real>());
      Real const cta_angle_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              angle_score, shared, mgpu::plus_t<Real>());
      Real const cta_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              torsion_score, shared, mgpu::plus_t<Real>());
      Real const cta_improper_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              improper_torsion_score, shared, mgpu::plus_t<Real>());
      Real const cta_hxyl_torsion_score =
          DeviceDispatch<D>::template reduce_in_workgroup<nt>(
              hxyl_torsion_score, shared, mgpu::plus_t<Real>());

      if (tid == 0) {
        int const output_index = (output_block_pair_energies) ? cta : pose_ind;
        if (cta_length_score != 0.0) {
          accumulate<D, Real>::add(V[0][output_index], cta_length_score);
        }
        if (cta_angle_score != 0.0) {
          accumulate<D, Real>::add(V[1][output_index], cta_angle_score);
        }
        if (cta_torsion_score != 0.0) {
          accumulate<D, Real>::add(V[2][output_index], cta_torsion_score);
        }
        if (cta_improper_torsion_score != 0.0) {
          accumulate<D, Real>::add(
              V[3][output_index], cta_improper_torsion_score);
        }
        if (cta_hxyl_torsion_score != 0.0) {
          accumulate<D, Real>::add(V[4][output_index], cta_hxyl_torsion_score);
        }
      }
    });
    DeviceDispatch<D>::template for_each_in_workgroup<nt>(reduce_energies);
  });
  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_subgraphs_for_interaction);

  return {
      V_t,
      dV_dx_t,
      dispatch_indices_t,
      n_output_intxns_for_rot_conn_offset_t,
      rotconn_for_output_intxn_t,
  };
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedRotamerScoreDispatch<DeviceDispatch, D, Real, Int>::backward(
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
    ) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);

  int const max_n_conns = pose_stack_inter_block_connections.size(2);
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
  assert(pose_stack_inter_block_connections.size(2) == max_n_conns);

  assert(atom_paths_from_conn.size(0) == n_block_types);
  assert(atom_paths_from_conn.size(1) == max_n_conns);
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
  // Define nt and reduce_t
  CTA_REAL_REDUCE_T_TYPEDEF;

  int const n_output_intxns_total = dispatch_indices.size(1);
  // We only call backward if we are in output_block_pair_energies mode,
  // so we will just go directly to allocating V_t w/ n_intxns_total size
  // int const n_V = output_block_pair_energies ? n_intxns_total : n_poses;
  auto V_t = TPack<Real, 2, D>::zeros({5, n_output_intxns_total});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({5, n_atoms});
  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  auto eval_subgraphs_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const rotconn_ind = rotconn_for_output_intxn[cta];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real length_score = 0.0;
    Real angle_score = 0.0;
    Real torsion_score = 0.0;
    Real improper_torsion_score = 0.0;
    Real hxyl_torsion_score = 0.0;

    auto score_subgraph =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
          Vec<Real, 7> params = hash_values[param_index];

          Vec<Real, 3> atom1 = rot_coords[atoms[0]];
          Vec<Real, 3> atom2 = rot_coords[atoms[1]];

          subgraph_type type = get_subgraph_type(atoms);

          int score_type = params[0];
          Real block_weight = dTdV[score_type][cta];

          // Real score;
          switch (type) {
            case subgraph_type::length: {
              auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
              accumulate_result<Real, Int, 2, D>(
                  length_score,
                  eval,
                  atoms.head(2),
                  true,
                  dV_dx[score_type],
                  block_weight);

              break;
            }
            case subgraph_type::angle: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              auto eval =
                  cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
              accumulate_result<Real, Int, 3, D>(
                  angle_score,
                  eval,
                  atoms.head(3),
                  true,
                  dV_dx[score_type],
                  block_weight);

              break;
            }
            case subgraph_type::torsion: {
              Vec<Real, 3> atom3 = rot_coords[atoms[2]];
              Vec<Real, 3> atom4 = rot_coords[atoms[3]];
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
              Real& tor_score = (score_type == 3)   ? improper_torsion_score
                                : (score_type == 4) ? hxyl_torsion_score
                                                    : torsion_score;
              accumulate_result<Real, Int, 4, D>(
                  tor_score,
                  eval,
                  atoms.head(4),
                  true,
                  dV_dx[score_type],
                  block_weight);

              break;
            }
          }
        });

    if (conn_ind1 == max_n_conns) {
      // intra-residue!
      int subgraph_offset = cart_subgraph_offsets[block_type1];
      int subgraph_offset_next = block_type1 + 1 == n_block_types
                                     ? n_subgraphs
                                     : cart_subgraph_offsets[block_type1 + 1];
      int n_subgraphs = subgraph_offset_next - subgraph_offset;
      auto eval_intra_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_subgraphs; i += nt) {
          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};
          for (bool reverse : {false, true}) {
            Vec<Int, 4> subgraph = cart_subgraphs[subgraph_offset + i];
            if (reverse) reverse_subgraph(subgraph);

            Vec<Int, 4> subgraph_atom_ids =
                get_atom_ids(atom_unique_ids[block_type1], subgraph);
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

            subgraph_atom_indices =
                atom_local_to_global_indices(subgraph, rot_coord_offset1);

            if (param_index != -1) {
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_intra_res_subgraphs);
    } else {
      // Inter residue!
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      int const local_rot_ind1 =
          rot_ind1 - rot_offset_for_block[pose_ind][block_ind1];
      int const local_rot_ind2 =
          cta - n_output_intxns_for_rot_conn_offset[rotconn_ind];
      int const rot_ind2 =
          rot_offset_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // Use by-reference capture here so that we can write to the length,
      // angle, and torsion_score variables
      auto eval_inter_res_subgraphs = ([&] TMOL_DEVICE_FUNC(int tid) {
        int n_connection_spanning_subgraphs = common::NUM_INTER_RES_PATHS;
        for (int i = tid; i < 2 * n_connection_spanning_subgraphs; i += nt) {
          bool reverse = i % 2 == 1;
          // interleave the subgraphs from the two directions so we can have
          // better warp coherence.
          int const subgraph_index = i / 2;
          int const block_typeA = reverse ? block_type2 : block_type1;
          int const block_typeB = reverse ? block_type1 : block_type2;
          int const rot_coord_offsetA =
              reverse ? rot_coord_offset2 : rot_coord_offset1;
          int const rot_coord_offsetB =
              reverse ? rot_coord_offset1 : rot_coord_offset2;
          int const conn_indA = reverse ? conn_ind2 : conn_ind1;
          int const conn_indB = reverse ? conn_ind1 : conn_ind2;

          int param_index = -1;
          Vec<Int, 4> subgraph_atom_indices = {-1, -1, -1, -1};

          tuple<int, int, int> spanning_subgraphs =
              get_connection_spanning_subgraph_indices(subgraph_index);
          int resA_path_ind = common::get<1>(spanning_subgraphs);
          int resB_path_ind = common::get<2>(spanning_subgraphs);
          // Grab the paths from each block
          Vec<Int, 3> resA_path =
              atom_paths_from_conn[block_typeA][conn_indA][resA_path_ind];
          Vec<Int, 3> resB_path =
              atom_paths_from_conn[block_typeB][conn_indB][resB_path_ind];

          // Make sure these are valid paths
          if (resA_path[0] == -1 || resB_path[0] == -1) continue;
          // Reverse the first path so that we can join them head-to-head
          resA_path.reverseInPlace();
          // Get a new Vec containing the global indices of the atoms
          Vec<Int, 3> resA_atom_indices =
              atom_local_to_global_indices(resA_path, rot_coord_offsetA);
          Vec<Int, 3> resB_atom_indices =
              atom_local_to_global_indices(resB_path, rot_coord_offsetB);
          // Calculate the size of each path
          Int resA_size = (resA_atom_indices.array() != -1).count();
          Int resB_size = (resB_atom_indices.array() != -1).count();

          // Try both unique and wildcard IDs for block A
          for (bool wildcard : {false, true}) {
            // Get the lookup tables for atom ID
            const auto& resA_atom_id_table =
                (wildcard) ? atom_wildcard_ids[block_typeA]
                           : atom_unique_ids[block_typeA];
            const auto& resB_atom_id_table = atom_wildcard_ids[block_typeB];

            // Get the atom IDs
            Vec<Int, 3> resA_subgraph_atom_ids =
                get_atom_ids(resA_atom_id_table, resA_path);
            Vec<Int, 3> resB_subgraph_atom_ids =
                get_atom_ids(resB_atom_id_table, resB_path);

            // Make the joined data structures
            Vec<Int, 4> path;
            Vec<Int, 4> atom_indices;
            Vec<Int, 4> subgraph_atom_ids;

            // Init with -1s
            path << -1, -1, -1, -1;
            atom_indices << -1, -1, -1, -1;
            subgraph_atom_ids << -1, -1, -1, -1;

            // Join the paths into 1
            path.head(resA_size + resB_size) << resA_path.tail(resA_size),
                resB_path.head(resB_size);
            atom_indices.head(resA_size + resB_size)
                << resA_atom_indices.tail(resA_size),
                resB_atom_indices.head(resB_size);
            subgraph_atom_ids.head(resA_size + resB_size)
                << resA_subgraph_atom_ids.tail(resA_size),
                resB_subgraph_atom_ids.head(resB_size);

            // Do the lookup
            param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);
            if (param_index != -1) {
              // we found it!
              subgraph_atom_indices = atom_indices;
              break;
            }
          }
          if (param_index != -1) {
            score_subgraph(subgraph_atom_indices, param_index);
          }
        }
      });
      DeviceDispatch<D>::template for_each_in_workgroup<nt>(
          eval_inter_res_subgraphs);
    }
  });

  DeviceDispatch<D>::template foreach_workgroup<launch_t>(
      dispatch_indices.size(1), eval_subgraphs_for_interaction);

  return dV_dx_t;

}  // backward

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
