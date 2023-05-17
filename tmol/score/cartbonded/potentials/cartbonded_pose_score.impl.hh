#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
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

enum class subgraph_type { length, angle, torsion };

template <typename Int>
TMOL_DEVICE_FUNC subgraph_type get_subgraph_type(Vec<Int, 4> subgraph) {
  if (subgraph[2] == -1 && subgraph[3] == -1) return subgraph_type::length;
  if (subgraph[3] == -1) return subgraph_type::angle;
  return subgraph_type::torsion;
}

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
    TView<Vec<Real, 6>, 1, D> hash_values,
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

  auto V_t = TPack<Real, 2, D>::zeros({3, n_poses});
  auto dV_dx_t = TPack<Vec<Real, 3>, 3, D>::zeros({3, n_poses, max_n_atoms});

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
    const int CON_PATH_INDICES[][2] = {// Length
                                       {0, 0},

                                       // Angles
                                       {0, 1},
                                       {0, 2},
                                       {0, 3},
                                       {1, 0},
                                       {2, 0},
                                       {3, 0},

                                       // Torsions
                                       {0, 4},
                                       {0, 5},
                                       {0, 6},
                                       {0, 7},
                                       {0, 8},
                                       {0, 9},
                                       {0, 10},
                                       {0, 11},
                                       {0, 12},

                                       {1, 1},
                                       {1, 2},
                                       {1, 3},
                                       {2, 1},
                                       {2, 2},
                                       {2, 3},
                                       {3, 1},
                                       {3, 2},
                                       {3, 3},

                                       {4, 0},
                                       {5, 0},
                                       {6, 0},
                                       {7, 0},
                                       {8, 0},
                                       {9, 0},
                                       {10, 0},
                                       {11, 0},
                                       {12, 0}};

    int block_type = pose_stack_block_type[pose_index][block_index];
    auto pose_coords = coords[pose_index];
    int block_coord_offset =
        pose_stack_block_coord_offset[pose_index][block_index];
    int subgraph_offset = cart_subgraph_offsets[block_type];
    int subgraph_offset_next = block_type + 1 == cart_subgraph_offsets.size(0)
                                   ? cart_subgraphs.size(0)
                                   : cart_subgraph_offsets[block_type + 1];
    subgraph_index += subgraph_offset;

    auto get_block_type_atom_id =
        ([=] TMOL_DEVICE_FUNC(
             Int atom_index, Int block_type_index, bool wildcard = false) {
          return (atom_index == -1) ? -1
                 : (wildcard) ? atom_wildcard_ids[block_type_index][atom_index]
                              : atom_unique_ids[block_type_index][atom_index];
        });

    auto get_param_index =
        ([=] TMOL_DEVICE_FUNC(Vec<Int, 4> subgraph_atom_ids) {
          int index = hash_lookup<Int, 4, D>(subgraph_atom_ids, hash_keys);

          return index;
        });

    auto score_subgraph =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Int param_index) {
          Vec<Real, 6> params = hash_values[param_index];

          Vec<Real, 3> atom1 = pose_coords[atoms[0]];
          Vec<Real, 3> atom2 = pose_coords[atoms[1]];
          Vec<Real, 3> atom3 = pose_coords[atoms[2]];
          Vec<Real, 3> atom4 = pose_coords[atoms[3]];

          subgraph_type type = get_subgraph_type(atoms);

          // Real score;
          switch (type) {
            case subgraph_type::length: {
              auto eval = cblength_V_dV(atom1, atom2, params[1], params[0]);
              score = common::get<0>(eval);
              accumulate<D, Real>::add(V[0][pose_index], common::get<0>(eval));
              break;
            }
            case subgraph_type::angle: {
              auto eval =
                  cbangle_V_dV(atom1, atom2, atom3, params[1], params[0]);
              score = common::get<0>(eval);
              accumulate<D, Real>::add(V[1][pose_index], common::get<0>(eval));
              break;
            }
            case subgraph_type::torsion: {
              auto eval = cbhxltorsion_V_dV(
                  atom1,
                  atom2,
                  atom3,
                  atom4,
                  params[0],
                  params[1],
                  params[2],
                  params[3],
                  params[4],
                  params[5]);
              score = common::get<0>(eval);
              accumulate<D, Real>::add(V[2][pose_index], common::get<0>(eval));
              break;
            }
          }
        });

    if (subgraph_index >= subgraph_offset_next) {
      if (subgraph_index >= subgraph_offset_next + NUM_INTER_RES_PATHS) return;

      subgraph_index -= subgraph_offset_next;

      for (int i = 0; i < pose_stack_inter_block_connections.size(2); i++) {
        for (bool wildcard : {false, true}) {
          const Vec<Int, 2>& connection =
              pose_stack_inter_block_connections[pose_index][block_index][i];
          int other_block_index = connection[0];
          int other_block_type =
              pose_stack_block_type[pose_index][other_block_index];

          if (other_block_index == -1) continue;

          int other_connection_index = connection[1];

          int res1_path_ind = CON_PATH_INDICES[subgraph_index][0];
          int res2_path_ind = CON_PATH_INDICES[subgraph_index][1];

          Vec<Int, 3> res1_path =
              atom_paths_from_conn[block_type][i][res1_path_ind];
          Vec<Int, 3> res2_path =
              atom_paths_from_conn[other_block_type][other_connection_index]
                                  [res2_path_ind];

          if (res1_path[0] == -1 || res2_path[0] == -1) continue;
          res1_path.reverseInPlace();

          Vec<Int, 3> res1_atom_indices;
          Vec<Int, 3> res2_atom_indices;

          Vec<Int, 3> res1_subgraph_atom_ids;
          Vec<Int, 3> res2_subgraph_atom_ids;

          Int res1_size = 0;
          Int res2_size = 0;

          // Fill the other vecs with additional data and mark the start/end
          for (int j = 0; j < 3; j++) {
            if (res1_path[j] != -1) {
              res1_atom_indices[j] = res1_path[j] + block_coord_offset;
              res1_subgraph_atom_ids[j] =
                  get_block_type_atom_id(res1_path[j], block_type, wildcard);
              res1_size++;
            } else {
              res1_atom_indices[j] = -1;
              res1_subgraph_atom_ids[j] = -1;
            }

            if (res2_path[j] != -1) {
              res2_atom_indices[j] =
                  res2_path[j]
                  + pose_stack_block_coord_offset[pose_index]
                                                 [other_block_index];
              res2_subgraph_atom_ids[j] =
                  get_block_type_atom_id(res2_path[j], other_block_type, true);
              res2_size++;
            } else {
              res2_atom_indices[j] = -1;
              res2_subgraph_atom_ids[j] = -1;
            }
          }

          Vec<Int, 4> path;
          Vec<Int, 4> atom_indices;
          Vec<Int, 4> subgraph_atom_ids;

          path << -1, -1, -1, -1;
          atom_indices << -1, -1, -1, -1;
          subgraph_atom_ids << -1, -1, -1, -1;

          path.head(res1_size + res2_size) << res1_path.tail(res1_size),
              res2_path.head(res2_size);
          atom_indices.head(res1_size + res2_size)
              << res1_atom_indices.tail(res1_size),
              res2_atom_indices.head(res2_size);
          subgraph_atom_ids.head(res1_size + res2_size)
              << res1_subgraph_atom_ids.tail(res1_size),
              res2_subgraph_atom_ids.head(res2_size);

          int param_index = get_param_index(subgraph_atom_ids);

          if (param_index != -1) {
            score_subgraph(atom_indices, param_index);
          }
        }
      }
      return;
    }

    for (bool reverse : {false, true}) {
      Vec<Int, 4> subgraph = cart_subgraphs[subgraph_index];
      if (reverse) reverse_subgraph(subgraph);

      Vec<Int, 4> subgraph_atom_ids;
      for (int i = 0; i < 4; i++) {
        subgraph_atom_ids[i] = get_block_type_atom_id(subgraph[i], block_type);
      }
      int param_index = get_param_index(subgraph_atom_ids);

      Vec<Int, 4> subgraph_atom_indices;
      for (int i = 0; i < 4; i++)
        subgraph_atom_indices[i] =
            (subgraph[i] == -1) ? -1 : block_coord_offset + subgraph[i];
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
