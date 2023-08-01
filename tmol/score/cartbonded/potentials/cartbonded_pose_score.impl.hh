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
    atom_ids[i] = get_atom_id<Int, D>(atom_id_table, atoms[i]);
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
    TensorAccessor<Vec<Real, 3>, 1, D> dV) {
  accumulate<D, Real>::add(V, common::get<0>(to_add));
  for (int i = 0; i < N; i++) {
    accumulate<D, Vec<Real, 3>>::add(dV[atoms[i]], common::get<1>(to_add)[i]);
  }
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto CartBondedPoseScoreDispatch<DeviceDispatch, D, Real, Int>::f(
    TView<Vec<Real, 3>, 2, D> coords,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Vec<Int, 3>, 3, D> atom_paths_from_conn,
    TView<Int, 2, D> atom_unique_ids,
    TView<Int, 2, D> atom_wildcard_ids,
    TView<Vec<Int, 5>, 1, D> hash_keys,
    TView<Vec<Real, 7>, 1, D> hash_values,
    TView<Vec<Int, 4>, 1, D> cart_subgraphs,
    TView<Int, 1, D> cart_subgraph_offsets,

    int max_subgraphs_per_block,

    // TView<CartBondedGlobalParams<Real>, 1, D> global_params,

    bool compute_derivs

    ) -> std::tuple<TPack<Real, 2, D>, TPack<Vec<Real, 3>, 3, D>> {
  int const n_poses = coords.size(0);
  int const n_blocks = pose_stack_block_coord_offset.size(1);
  int const max_n_atoms = coords.size(1);
  int const NUM_INTER_RES_PATHS = 34;

  int const n_subgraphs = cart_subgraphs.size(0);

  auto V_t = TPack<Real, 2, D>::zeros({5, n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({5, n_poses, max_n_atoms});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  int const n_block_types = cart_subgraph_offsets.size(0);

  max_subgraphs_per_block +=
      NUM_INTER_RES_PATHS;  // Add in the inter-residue subgraphs

  // Optimal launch box on v100 and a100 is nt=32, vt=1
  LAUNCH_BOX_32;

  auto func = ([=] TMOL_DEVICE_FUNC(
                   int pose_index, int block_index, int subgraph_index) {
    Real score = 0;

    int block_type = pose_stack_block_type[pose_index][block_index];
    auto pose_coords = coords[pose_index];
    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][block_index];
    int subgraph_offset = cart_subgraph_offsets[block_type];
    int subgraph_offset_next = block_type + 1 == cart_subgraph_offsets.size(0)
                                   ? cart_subgraphs.size(0)
                                   : cart_subgraph_offsets[block_type + 1];
    subgraph_index += subgraph_offset;

    auto score_subgraph =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
          Vec<Real, 7> params = hash_values[param_index];

          Vec<Real, 3> atom1 = pose_coords[atoms[0]];
          Vec<Real, 3> atom2 = pose_coords[atoms[1]];

          subgraph_type type = get_subgraph_type(atoms);

          int score_type = params[0];

          // Real score;
          switch (type) {
            case subgraph_type::length: {
              auto eval = cblength_V_dV(atom1, atom2, params[2], params[1]);
              accumulate_result<Real, Int, 2, D>(
                  eval,
                  atoms.head(2),
                  V[score_type][pose_index],
                  dV_dx[score_type][pose_index]);

              break;
            }
            case subgraph_type::angle: {
              Vec<Real, 3> atom3 = pose_coords[atoms[2]];
              auto eval =
                  cbangle_V_dV(atom1, atom2, atom3, params[2], params[1]);
              accumulate_result<Real, Int, 3, D>(
                  eval,
                  atoms.head(3),
                  V[score_type][pose_index],
                  dV_dx[score_type][pose_index]);

              break;
            }
            case subgraph_type::torsion: {
              Vec<Real, 3> atom3 = pose_coords[atoms[2]];
              Vec<Real, 3> atom4 = pose_coords[atoms[3]];
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
                  atoms.head(4),
                  V[score_type][pose_index],
                  dV_dx[score_type][pose_index]);

              break;
            }
          }
        });

    if (subgraph_index >= subgraph_offset_next) {
      if (subgraph_index >= subgraph_offset_next + NUM_INTER_RES_PATHS) return;

      subgraph_index -= subgraph_offset_next;

      // Iterate over each connection in this block
      for (int i = 0; i < pose_stack_inter_block_connections.size(2); i++) {
        const Vec<Int, 2>& connection =
            pose_stack_inter_block_connections[pose_index][block_index][i];
        int other_block_index = connection[0];
        int other_block_type =
            pose_stack_block_type[pose_index][other_block_index];
        int other_block_offset =
            pose_stack_block_coord_offset[pose_index][other_block_index];

        // No block on the other side of the connection, nothing to do
        if (other_block_index == -1) continue;

        int other_connection_index = connection[1];

        // From our subgraph index, grab the corresponding paths indices for
        // each block
        tuple<int, int> spanning_subgraphs =
            get_connection_spanning_subgraph_indices(subgraph_index);
        int res1_path_ind = common::get<0>(spanning_subgraphs);
        int res2_path_ind = common::get<1>(spanning_subgraphs);

        // Grab the paths from each block
        Vec<Int, 3> res1_path =
            atom_paths_from_conn[block_type][i][res1_path_ind];
        Vec<Int, 3> res2_path =
            atom_paths_from_conn[other_block_type][other_connection_index]
                                [res2_path_ind];

        // Make sure these are valid paths
        if (res1_path[0] == -1 || res2_path[0] == -1) continue;
        // Reverse the first path so that we can join them head-to-head
        res1_path.reverseInPlace();

        // Get a new Vec containing the global indices of the atoms
        Vec<Int, 3> res1_atom_indices =
            atom_local_to_global_indices(res1_path, block_coord_offset);
        Vec<Int, 3> res2_atom_indices =
            atom_local_to_global_indices(res2_path, other_block_offset);

        // Calculate the size of each path
        Int res1_size = (res1_atom_indices.array() != -1).count();
        Int res2_size = (res2_atom_indices.array() != -1).count();

        // Try both unique and wildcard IDs for block 1
        for (bool wildcard : {false, true}) {
          // Get the lookup tables for atom ID
          const auto& res1_atom_id_table = (wildcard)
                                               ? atom_wildcard_ids[block_type]
                                               : atom_unique_ids[block_type];
          const auto& res2_atom_id_table = atom_wildcard_ids[other_block_type];

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

          // If we found a param that matches, score the subgraph
          if (param_index != -1) {
            score_subgraph(atom_indices, param_index);
          }
        }
      }
      return;
    }

    // Intra-res subgraphs
    for (bool reverse : {false, true}) {
      Vec<Int, 4> subgraph = cart_subgraphs[subgraph_index];
      if (reverse) reverse_subgraph(subgraph);

      Vec<Int, 4> subgraph_atom_ids =
          get_atom_ids(atom_unique_ids[block_type], subgraph);
      int param_index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

      Vec<Int, 4> subgraph_atom_indices =
          atom_local_to_global_indices(subgraph, block_coord_offset);

      if (param_index != -1) {
        score_subgraph(subgraph_atom_indices, param_index);
      }
    }
  });

  DeviceDispatch<D>::foreach_combination_triple(
      n_poses, n_blocks, max_subgraphs_per_block, func);

  return {V_t, dV_dx_t};
}  // namespace potentials

}  // namespace potentials
}  // namespace cartbonded
}  // namespace score
}  // namespace tmol
