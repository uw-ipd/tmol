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
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/score/common/geom.hh>
#include <tmol/score/common/hash_util.hh>
#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/warp_segreduce.hh>
#include <tmol/score/common/warp_stride_reduce.hh>

#include "params.hh"
#include "potentials.hh"
#include "genbonded_pose_score.hh"

// mgpu operators for warp/CTA reductions
#include <moderngpu/operators.hxx>

namespace tmol {
namespace score {
namespace genbonded {
namespace potentials {

using namespace tmol::score::common;

template <typename Real, int N>
using Vec = Eigen::Matrix<Real, N, 1>;

#define Real3 Vec<Real, 3>

// Maximum hierarchy depth stored per atom in gen_atom_type_hierarchy.
// Must match MAX_HIER_DEPTH in genbonded_energy_term.py.
#define GB_MAX_HIER_DEPTH 3

// Bond-type encoding for the central bond in inter-block torsion hash keys.
// 0 = wildcard ('~') — matches any bond type.
// Must match BOND_CHAR_TO_INT["~"] in genbonded_energy_term.py.
#define GB_BOND_WILDCARD 0

// ---------------------------------------------------------------------------
// Helper: convert block-local atom indices to global rot_coord indices.
// Entries that are -1 (sentinels for missing atoms) are preserved as -1.
// ---------------------------------------------------------------------------
template <typename Int, Int size>
TMOL_DEVICE_FUNC Vec<Int, size> atom_local_to_global_indices(
    Vec<Int, size> local_indices, Int offset) {
  Vec<Int, size> global_indices;
  for (int i = 0; i < size; i++) {
    global_indices[i] =
        (local_indices[i] != -1) ? local_indices[i] + offset : -1;
  }
  return global_indices;
}

// ---------------------------------------------------------------------------
// accumulate_torsion_result
//
// Mirrors cartbonded's accumulate_result but specialised for the
// tuple<Real, Vec<Real3,4>> returned by gbtorsion_V_dV.
// ---------------------------------------------------------------------------
template <typename Real, typename Int, tmol::Device D>
TMOL_DEVICE_FUNC void accumulate_torsion_result(
    Real& val,
    common::tuple<Real, Vec<Real3, 4>> to_add,
    Vec<Int, 4> atoms,
    bool accumulate_derivs,
    TensorAccessor<Vec<Real, 3>, 1, D> dV,
    const Real& weight = 1.0) {
  val += common::get<0>(to_add);
  if (accumulate_derivs) {
    for (int i = 0; i < 4; i++) {
      accumulate<D, Vec<Real, 3>>::add(
          dV[atoms[i]], common::get<1>(to_add)[i] * weight);
    }
  }
}

// ---------------------------------------------------------------------------
// inter_block_torsion_lookup
//
// Given atom-type hierarchy indices for 4 atoms (each a Vec<Int,3> of
// hierarchy levels from most specific to least specific), and the bond type
// integer for the central bond, try all GB_MAX_HIER_DEPTH^4 combinations and
// return the best (lowest total hierarchy score) matching entry.
//
// For each combination, the specific bond type is tried first; if not found
// and bond_type_int is not already wildcard (GB_BOND_WILDCARD=0), the lookup
// is retried with the wildcard key.
//
// hash_keys layout: Vec<Int,6> = [t1, t2, t3, t4, bond_type, val_idx]
// This matches hash_lookup<Int, 5, D>.
//
// Returns val_idx >= 0 if a match was found, -1 otherwise.
// ---------------------------------------------------------------------------
template <typename Int, tmol::Device D>
TMOL_DEVICE_FUNC int inter_block_torsion_lookup(
    Vec<Int, GB_MAX_HIER_DEPTH> h1,
    Vec<Int, GB_MAX_HIER_DEPTH> h2,
    Vec<Int, GB_MAX_HIER_DEPTH> h3,
    Vec<Int, GB_MAX_HIER_DEPTH> h4,
    Int bond_type_int,
    TView<Vec<Int, 6>, 1, D> hash_keys) {
  int best_val_idx = -1;
  int best_score = 999;

  for (int i = 0; i < GB_MAX_HIER_DEPTH && h1[i] != -1; i++) {
    for (int j = 0; j < GB_MAX_HIER_DEPTH && h2[j] != -1; j++) {
      for (int k = 0; k < GB_MAX_HIER_DEPTH && h3[k] != -1; k++) {
        for (int l = 0; l < GB_MAX_HIER_DEPTH && h4[l] != -1; l++) {
          int const score = i + j + k + l;
          if (score >= best_score) continue;

          Vec<Int, 5> key;
          key[0] = h1[i];
          key[1] = h2[j];
          key[2] = h3[k];
          key[3] = h4[l];
          key[4] = bond_type_int;

          int val_idx = hash_lookup<Int, 5, D>(key, hash_keys);

          // If specific bond type not found, retry with wildcard.
          if (val_idx < 0 && bond_type_int != GB_BOND_WILDCARD) {
            key[4] = GB_BOND_WILDCARD;
            val_idx = hash_lookup<Int, 5, D>(key, hash_keys);
          }

          if (val_idx >= 0) {
            best_score = score;
            best_val_idx = val_idx;
          }
        }
      }
    }
  }
  return best_val_idx;
}

// ===========================================================================
// GenBondedPoseScoreDispatch::forward
// ===========================================================================
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedPoseScoreDispatch<DeviceOps, D, Real, Int>::forward(
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
    -> std::tuple<TPack<Real, 4, D>, TPack<Vec<Real, 3>, 2, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = gen_intra_subgraph_offsets.size(0);
  int const n_total_intra = gen_intra_subgraphs.size(0);

  // One score type: gen_torsions (index 0).
  int const n_V = output_block_pair_energies ? max_n_blocks : 1;
  auto V_t = TPack<Real, 4, D>::zeros({1, n_poses, n_V, n_V});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;

  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_torsions_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      int stub;
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    // Decode CTA index: (pose, block, connection).
    int const pose_ind = cta / (max_n_blocks * (max_n_conns + 1));
    int const block_conn = cta % (max_n_blocks * (max_n_conns + 1));
    int const block_ind1 = block_conn / (max_n_conns + 1);
    int const conn_ind1 = block_conn % (max_n_conns + 1);
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];

    if (block_type1 == -1) return;

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real score = 0.0;

    // -----------------------------------------------------------------------
    // score_torsion: evaluate gbtorsion_V_dV for one 4-atom subgraph.
    // -----------------------------------------------------------------------
    auto score_torsion =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          auto eval = gbtorsion_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],  // k1
              prm[1],  // k2
              prm[2],  // k3
              prm[3],  // k4
              prm[4]   // offset
          );
          accumulate_torsion_result<Real, Int, D>(
              score, eval, atoms, compute_derivs, dV_dx[0], 1.0);
        });

    // -----------------------------------------------------------------------
    // score_improper: evaluate gbimproper_V_dV for one 4-atom subgraph.
    // prm: Vec<Real,5> with prm[0]=k, prm[1]=delta (prm[2..4] unused).
    // -----------------------------------------------------------------------
    auto score_improper =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          auto eval = gbimproper_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],  // k
              prm[1]   // delta
          );
          accumulate_torsion_result<Real, Int, D>(
              score, eval, atoms, compute_derivs, dV_dx[0], 1.0);
        });

    int block_ind2 = -1;

    if (conn_ind1 == max_n_conns) {
      // -----------------------------------------------------------------------
      // INTRA-BLOCK: combined proper + improper torsions.
      // Each entry: sg = Vec<Int,5> {tag, a0, a1, a2, a3}
      //   tag == 0 → proper torsion
      //   tag == 1 → improper torsion
      // -----------------------------------------------------------------------
      block_ind2 = block_ind1;

      int const intra_start = gen_intra_subgraph_offsets[block_type1];
      int const intra_end = (block_type1 + 1 == n_block_types)
                                ? n_total_intra
                                : gen_intra_subgraph_offsets[block_type1 + 1];
      int const n_intra = intra_end - intra_start;

      auto eval_intra = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_intra; i += nt) {
          int const idx = intra_start + i;
          Vec<Int, 5> sg = gen_intra_subgraphs[idx];
          if (sg[0] == -1) continue;

          // Extract the 4 local atom indices.
          Vec<Int, 4> local_sg;
          local_sg[0] = sg[1];
          local_sg[1] = sg[2];
          local_sg[2] = sg[3];
          local_sg[3] = sg[4];

          Vec<Int, 4> atoms =
              atom_local_to_global_indices(local_sg, rot_coord_offset1);
          Vec<Real, 5> prm = gen_intra_params[idx];

          if (sg[0] == 0) {
            score_torsion(atoms, prm);
          } else {
            score_improper(atoms, prm);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_intra);

    } else {
      // -----------------------------------------------------------------------
      // INTER-BLOCK torsions (spanning connection conn_ind1 of block_ind1).
      //
      // Algorithm mirrors cartbonded:
      //   1. Find the connected block (block_ind2) via inter-block connections.
      //   2. Lower-indexed block handles the interaction (no double-counting).
      //   3. Walk atom_paths_from_conn on both sides to enumerate 4-atom paths.
      //   4. For each path, look up torsion params via the chemical-type hash
      //      table using inter_block_torsion_lookup (hierarchy walk + bond
      //      type).
      // -----------------------------------------------------------------------
      block_ind2 = pose_stack_inter_block_connections[pose_ind][block_ind1]
                                                     [conn_ind1][0];
      if (block_ind2 == -1) return;
      // Lower block handles interaction to avoid double-counting.
      if (block_ind1 > block_ind2) return;

      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];

      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      // Bond type of the central inter-block bond (from block1's connection).
      Int const bond_type_int =
          gen_connection_bond_types[block_type1][conn_ind1];

      auto eval_inter_block = ([&] TMOL_DEVICE_FUNC(int tid) {
        // atom_paths_from_conn convention (same as cartbonded):
        //   path[0] = atom AT the connection (sentinel: -1 iff invalid path)
        //   path[1] = atom 1 hop from connection  (-1 if path has only 1 atom)
        //   path[2] = atom 2 hops from connection (-1 if path has <= 2 atoms)
        //
        // Assembly mirrors cartbonded:
        //   1. Reverse pathA (conn→...→distal becomes distal→...→conn).
        //   2. Join: pathA_reversed.tail(lenA) ++ pathB.head(lenB).
        //   Resulting 4-atom sequence: ...A_distal, A_conn, B_conn, B_distal...

        int const n_pairs = MAX_PATHS_FROM_CONN * MAX_PATHS_FROM_CONN;

        for (int pair = tid; pair < n_pairs; pair += nt) {
          int const path_A_idx = pair / MAX_PATHS_FROM_CONN;
          int const path_B_idx = pair % MAX_PATHS_FROM_CONN;

          Vec<Int, 3> pathA =
              atom_paths_from_conn[block_type1][conn_ind1][path_A_idx];
          Vec<Int, 3> pathB =
              atom_paths_from_conn[block_type2][conn_ind2][path_B_idx];

          // path[0] == -1 → no connection atom → completely invalid path.
          if (pathA[0] == -1 || pathB[0] == -1) continue;

          // Reverse pathA: path[0]=conn becomes path[2], path[2]=distal becomes
          // path[0].
          Vec<Int, 3> pathA_rev;
          pathA_rev[0] = pathA[2];
          pathA_rev[1] = pathA[1];
          pathA_rev[2] = pathA[0];

          // Global coordinate indices.
          Vec<Int, 3> globalA =
              atom_local_to_global_indices(pathA_rev, rot_coord_offset1);
          Vec<Int, 3> globalB =
              atom_local_to_global_indices(pathB, rot_coord_offset2);

          // Count valid (non -1) atoms on each side.
          int lenA = 0;
          for (int pi = 0; pi < 3; pi++) {
            if (globalA[pi] != -1) lenA++;
          }
          int lenB = 0;
          for (int pi = 0; pi < 3; pi++) {
            if (globalB[pi] != -1) lenB++;
          }

          if (lenA + lenB != 4) continue;

          // Assemble the 4-atom global index array:
          //   globalA.tail(lenA) ++ globalB.head(lenB)
          Vec<Int, 4> atoms;
          atoms[0] = -1;
          atoms[1] = -1;
          atoms[2] = -1;
          atoms[3] = -1;
          for (int pi = 0; pi < lenA; pi++) {
            atoms[pi] = globalA[3 - lenA + pi];  // .tail(lenA)
          }
          for (int pi = 0; pi < lenB; pi++) {
            atoms[lenA + pi] = globalB[pi];  // .head(lenB)
          }

          // Assemble the LOCAL atom index array (same structure, needed for
          // hierarchy lookup which is indexed by block-local atom index).
          Vec<Int, 4> local_indices;
          local_indices[0] = -1;
          local_indices[1] = -1;
          local_indices[2] = -1;
          local_indices[3] = -1;
          for (int pi = 0; pi < lenA; pi++) {
            local_indices[pi] = pathA_rev[3 - lenA + pi];  // .tail(lenA)
          }
          for (int pi = 0; pi < lenB; pi++) {
            local_indices[lenA + pi] = pathB[pi];  // .head(lenB)
          }

          // Get hierarchy indices for each of the 4 atoms.
          Vec<Int, GB_MAX_HIER_DEPTH> h[4];
          for (int pos = 0; pos < 4; pos++) {
            int loc = local_indices[pos];
            int bt = (pos < lenA) ? block_type1 : block_type2;
            if (loc >= 0) {
              h[pos] = gen_atom_type_hierarchy[bt][loc];
            } else {
              h[pos] = Vec<Int, GB_MAX_HIER_DEPTH>::Constant(-1);
            }
          }

          int val_idx = inter_block_torsion_lookup<Int, D>(
              h[0],
              h[1],
              h[2],
              h[3],
              bond_type_int,
              gen_inter_torsion_hash_keys);

          if (val_idx >= 0) {
            Vec<Real, 5> prm = gen_inter_torsion_hash_values[val_idx];
            score_torsion(atoms, prm);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_inter_block);
    }

    // -------------------------------------------------------------------------
    // Reduce per-thread scores across the CTA and write to V.
    // -------------------------------------------------------------------------
    auto reduce_and_write = ([&] TMOL_DEVICE_FUNC(int tid) {
      Real const cta_score = DeviceOps<D>::template reduce_in_workgroup<nt>(
          score, shared, mgpu::plus_t<Real>());
      if (tid == 0 && cta_score != 0.0) {
        if (output_block_pair_energies) {
          V[0][pose_ind][block_ind1][block_ind2] = cta_score;
        } else {
          accumulate<D, Real>::add(V[0][pose_ind][0][0], cta_score);
        }
      }
    });
    DeviceOps<D>::template for_each_in_workgroup<nt>(reduce_and_write);
  });

  DeviceOps<D>::template foreach_workgroup<launch_t>(
      n_poses * max_n_blocks * (max_n_conns + 1),
      eval_torsions_for_interaction);

  return {V_t, dV_dx_t};
}

// ===========================================================================
// GenBondedPoseScoreDispatch::backward
// ===========================================================================
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedPoseScoreDispatch<DeviceOps, D, Real, Int>::backward(
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

    TView<Real, 4, D> dTdV) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = gen_intra_subgraph_offsets.size(0);
  int const n_total_intra = gen_intra_subgraphs.size(0);

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dx_t.view;

  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto eval_torsions_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const pose_ind = cta / (max_n_blocks * (max_n_conns + 1));
    int const block_conn = cta % (max_n_blocks * (max_n_conns + 1));
    int const block_ind1 = block_conn / (max_n_conns + 1);
    int const conn_ind1 = block_conn % (max_n_conns + 1);
    int const block_type1 = first_rot_block_type[pose_ind][block_ind1];

    if (block_type1 == -1) return;

    int const rot_ind1 = first_rot_for_block[pose_ind][block_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    // Weighted torsion gradient helper.
    auto score_torsion_weighted = ([=] TMOL_DEVICE_FUNC(
                                       Vec<Int, 4> atoms,
                                       Vec<Real, 5> prm,
                                       int bpose_ind,
                                       int bblock_ind1,
                                       int bblock_ind2) {
      Real const block_weight = dTdV[0][bpose_ind][bblock_ind1][bblock_ind2];
      auto eval = gbtorsion_V_dV(
          rot_coords[atoms[0]],
          rot_coords[atoms[1]],
          rot_coords[atoms[2]],
          rot_coords[atoms[3]],
          prm[0],
          prm[1],
          prm[2],
          prm[3],
          prm[4]);
      Real dummy = 0.0;
      accumulate_torsion_result<Real, Int, D>(
          dummy, eval, atoms, true, dV_dx[0], block_weight);
    });

    // Weighted improper gradient helper (prm[0]=k, prm[1]=delta).
    auto score_improper_weighted = ([=] TMOL_DEVICE_FUNC(
                                        Vec<Int, 4> atoms,
                                        Vec<Real, 5> prm,
                                        int bpose_ind,
                                        int bblock_ind1,
                                        int bblock_ind2) {
      Real const block_weight = dTdV[0][bpose_ind][bblock_ind1][bblock_ind2];
      auto eval = gbimproper_V_dV(
          rot_coords[atoms[0]],
          rot_coords[atoms[1]],
          rot_coords[atoms[2]],
          rot_coords[atoms[3]],
          prm[0],
          prm[1]);
      Real dummy = 0.0;
      accumulate_torsion_result<Real, Int, D>(
          dummy, eval, atoms, true, dV_dx[0], block_weight);
    });

    if (conn_ind1 == max_n_conns) {
      int const block_ind2 = block_ind1;

      // Combined intra-block loop (proper + improper).
      int const intra_start = gen_intra_subgraph_offsets[block_type1];
      int const intra_end = (block_type1 + 1 == n_block_types)
                                ? n_total_intra
                                : gen_intra_subgraph_offsets[block_type1 + 1];
      int const n_intra = intra_end - intra_start;

      auto eval_intra = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_intra; i += nt) {
          int const idx = intra_start + i;
          Vec<Int, 5> sg = gen_intra_subgraphs[idx];
          if (sg[0] == -1) continue;

          Vec<Int, 4> local_sg;
          local_sg[0] = sg[1];
          local_sg[1] = sg[2];
          local_sg[2] = sg[3];
          local_sg[3] = sg[4];

          Vec<Int, 4> atoms =
              atom_local_to_global_indices(local_sg, rot_coord_offset1);
          Vec<Real, 5> prm = gen_intra_params[idx];

          if (sg[0] == 0) {
            score_torsion_weighted(
                atoms, prm, pose_ind, block_ind1, block_ind2);
          } else {
            score_improper_weighted(
                atoms, prm, pose_ind, block_ind1, block_ind2);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_intra);

    } else {
      // Inter-block backward pass (mirrors forward inter-block logic exactly).
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      if (block_ind2 == -1) return;
      if (block_ind1 > block_ind2) return;

      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];

      int const rot_ind2 = first_rot_for_block[pose_ind][block_ind2];
      int const block_type2 = first_rot_block_type[pose_ind][block_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];

      Int const bond_type_int =
          gen_connection_bond_types[block_type1][conn_ind1];

      auto eval_inter_block = ([&] TMOL_DEVICE_FUNC(int tid) {
        // Mirrors forward eval_inter_block exactly (same path convention).
        int const n_pairs = MAX_PATHS_FROM_CONN * MAX_PATHS_FROM_CONN;

        for (int pair = tid; pair < n_pairs; pair += nt) {
          int const path_A_idx = pair / MAX_PATHS_FROM_CONN;
          int const path_B_idx = pair % MAX_PATHS_FROM_CONN;

          Vec<Int, 3> pathA =
              atom_paths_from_conn[block_type1][conn_ind1][path_A_idx];
          Vec<Int, 3> pathB =
              atom_paths_from_conn[block_type2][conn_ind2][path_B_idx];

          // path[0] == -1 → no connection atom → completely invalid path.
          if (pathA[0] == -1 || pathB[0] == -1) continue;

          // Reverse pathA: path[0]=conn becomes path[2], path[2]=distal becomes
          // path[0].
          Vec<Int, 3> pathA_rev;
          pathA_rev[0] = pathA[2];
          pathA_rev[1] = pathA[1];
          pathA_rev[2] = pathA[0];

          // Global coordinate indices.
          Vec<Int, 3> globalA =
              atom_local_to_global_indices(pathA_rev, rot_coord_offset1);
          Vec<Int, 3> globalB =
              atom_local_to_global_indices(pathB, rot_coord_offset2);

          // Count valid (non -1) atoms on each side.
          int lenA = 0;
          for (int pi = 0; pi < 3; pi++) {
            if (globalA[pi] != -1) lenA++;
          }
          int lenB = 0;
          for (int pi = 0; pi < 3; pi++) {
            if (globalB[pi] != -1) lenB++;
          }

          if (lenA + lenB != 4) continue;

          // Assemble the 4-atom global index array:
          //   globalA.tail(lenA) ++ globalB.head(lenB)
          Vec<Int, 4> atoms;
          atoms[0] = -1;
          atoms[1] = -1;
          atoms[2] = -1;
          atoms[3] = -1;
          for (int pi = 0; pi < lenA; pi++) {
            atoms[pi] = globalA[3 - lenA + pi];  // .tail(lenA)
          }
          for (int pi = 0; pi < lenB; pi++) {
            atoms[lenA + pi] = globalB[pi];  // .head(lenB)
          }

          // Assemble the LOCAL atom index array.
          Vec<Int, 4> local_indices;
          local_indices[0] = -1;
          local_indices[1] = -1;
          local_indices[2] = -1;
          local_indices[3] = -1;
          for (int pi = 0; pi < lenA; pi++) {
            local_indices[pi] = pathA_rev[3 - lenA + pi];  // .tail(lenA)
          }
          for (int pi = 0; pi < lenB; pi++) {
            local_indices[lenA + pi] = pathB[pi];  // .head(lenB)
          }

          // Get hierarchy indices for each of the 4 atoms.
          Vec<Int, GB_MAX_HIER_DEPTH> h[4];
          for (int pos = 0; pos < 4; pos++) {
            int loc = local_indices[pos];
            int bt = (pos < lenA) ? block_type1 : block_type2;
            if (loc >= 0) {
              h[pos] = gen_atom_type_hierarchy[bt][loc];
            } else {
              h[pos] = Vec<Int, GB_MAX_HIER_DEPTH>::Constant(-1);
            }
          }

          int val_idx = inter_block_torsion_lookup<Int, D>(
              h[0],
              h[1],
              h[2],
              h[3],
              bond_type_int,
              gen_inter_torsion_hash_keys);

          if (val_idx >= 0) {
            Vec<Real, 5> prm = gen_inter_torsion_hash_values[val_idx];
            score_torsion_weighted(
                atoms, prm, pose_ind, block_ind1, block_ind2);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_inter_block);
    }
    // No CTA reduction in backward — gradients accumulated directly via atomic
    // add.
  });

  DeviceOps<D>::template foreach_workgroup<launch_t>(
      n_poses * max_n_blocks * (max_n_conns + 1),
      eval_torsions_for_interaction);

  return dV_dx_t;
}

// ===========================================================================
// GenBondedRotamerScoreDispatch::forward
// ===========================================================================
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedRotamerScoreDispatch<DeviceOps, D, Real, Int>::forward(
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
        TPack<Real, 2, D>,
        TPack<Vec<Real, 3>, 2, D>,
        TPack<Int, 2, D>,
        TPack<Int, 1, D>,
        TPack<Int, 1, D>> {
  int const n_atoms = rot_coords.size(0);
  int const n_rots = rot_coord_offset.size(0);
  int const n_poses = first_rot_for_block.size(0);
  int const max_n_blocks = first_rot_for_block.size(1);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_block_types = gen_intra_subgraph_offsets.size(0);
  int const n_total_intra = gen_intra_subgraphs.size(0);

  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;

  // Convention: conn == max_n_conns => intra-rotamer interactions.
  int const max_n_interactions = n_rots * (max_n_conns + 1);

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
    if (block_type_ind == -1) return;

    if (conn_ind == max_n_conns) {
      n_output_intxns_for_rot_conn[index] = 1;
    } else {
      int const other_block_ind =
          pose_stack_inter_block_connections[pose_ind][block_ind][conn_ind][0];
      if (other_block_ind == -1) return;
      int const other_block_n_rots =
          n_rots_for_block[pose_ind][other_block_ind];
      if (block_ind < other_block_ind) {
        n_output_intxns_for_rot_conn[index] = other_block_n_rots;
      }
    }
  });
  DeviceOps<D>::template forall<launch_t>(
      max_n_interactions, count_intxns_for_rot_conn);

  int n_output_intxns_total =
      DeviceOps<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_output_intxns_for_rot_conn.data(),
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions,
          mgpu::plus_t<Int>());
  TPack<Int, 1, D> rotconn_for_output_intxn_t =
      DeviceOps<D>::template load_balancing_search<launch_t>(
          n_output_intxns_total,
          n_output_intxns_for_rot_conn_offset.data(),
          max_n_interactions);
  auto rotconn_for_output_intxn = rotconn_for_output_intxn_t.view;

  int const n_V = output_block_pair_energies ? n_output_intxns_total : n_poses;
  auto V_t = TPack<Real, 2, D>::zeros({1, n_V});
  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dispatch_indices_t = TPack<Int, 2, D>::zeros({3, n_output_intxns_total});

  auto V = V_t.view;
  auto dV_dx = dV_dx_t.view;
  auto dispatch_indices = dispatch_indices_t.view;

  auto record_dispatch_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rotconn_ind = rotconn_for_output_intxn[index];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    dispatch_indices[0][index] = pose_ind;
    dispatch_indices[1][index] = rot_ind1;
    int rot_ind2;
    if (conn_ind1 == max_n_conns) {
      rot_ind2 = rot_ind1;
    } else {
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const rotconn_offset =
          n_output_intxns_for_rot_conn_offset[rotconn_ind];
      int const local_rot_ind2 = index - rotconn_offset;
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      rot_ind2 = first_rot_for_block[pose_ind][block_ind2] + local_rot_ind2;
    }
    dispatch_indices[2][index] = rot_ind2;
  });
  DeviceOps<D>::template forall<launch_t>(
      n_output_intxns_total, record_dispatch_indices);

  auto eval_torsions_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    SHARED_MEMORY union shared_mem_union {
      shared_mem_union() {}
      int stub;
      CTA_REAL_REDUCE_T_VARIABLE;
    } shared;

    int const rotconn_ind = rotconn_for_output_intxn[cta];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];

    Real torsion_score = 0.0;

    auto score_torsion =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          auto eval = gbtorsion_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],
              prm[1],
              prm[2],
              prm[3],
              prm[4]);
          accumulate_torsion_result<Real, Int, D>(
              torsion_score, eval, atoms, compute_derivs, dV_dx[0], 1.0);
        });

    auto score_improper =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          auto eval = gbimproper_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],
              prm[1]);
          accumulate_torsion_result<Real, Int, D>(
              torsion_score, eval, atoms, compute_derivs, dV_dx[0], 1.0);
        });

    if (conn_ind1 == max_n_conns) {
      // Intra-block: iterate gen_intra_subgraphs with tag dispatch.
      int const intra_start = gen_intra_subgraph_offsets[block_type1];
      int const intra_end = (block_type1 + 1 == n_block_types)
                                ? n_total_intra
                                : gen_intra_subgraph_offsets[block_type1 + 1];
      int const n_intra = intra_end - intra_start;

      auto eval_intra = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_intra; i += nt) {
          int const idx = intra_start + i;
          Vec<Int, 5> sg = gen_intra_subgraphs[idx];
          if (sg[0] == -1) continue;
          Vec<Int, 4> local_sg;
          local_sg[0] = sg[1];
          local_sg[1] = sg[2];
          local_sg[2] = sg[3];
          local_sg[3] = sg[4];
          Vec<Int, 4> atoms =
              atom_local_to_global_indices(local_sg, rot_coord_offset1);
          Vec<Real, 5> prm = gen_intra_params[idx];
          if (sg[0] == 0) {
            score_torsion(atoms, prm);
          } else {
            score_improper(atoms, prm);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_intra);

    } else {
      // Inter-block: look up torsion params via hierarchy hash.
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      int const local_rot_ind2 =
          cta - n_output_intxns_for_rot_conn_offset[rotconn_ind];
      int const rot_ind2 =
          rot_offset_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];
      Int const bond_type_int =
          gen_connection_bond_types[block_type1][conn_ind1];

      auto eval_inter = ([&] TMOL_DEVICE_FUNC(int tid) {
        int const n_pairs = MAX_PATHS_FROM_CONN * MAX_PATHS_FROM_CONN;
        for (int pair = tid; pair < n_pairs; pair += nt) {
          int const path_A_idx = pair / MAX_PATHS_FROM_CONN;
          int const path_B_idx = pair % MAX_PATHS_FROM_CONN;
          Vec<Int, 3> pathA =
              atom_paths_from_conn[block_type1][conn_ind1][path_A_idx];
          Vec<Int, 3> pathB =
              atom_paths_from_conn[block_type2][conn_ind2][path_B_idx];
          if (pathA[0] == -1 || pathB[0] == -1) continue;

          Vec<Int, 3> pathA_rev;
          pathA_rev[0] = pathA[2];
          pathA_rev[1] = pathA[1];
          pathA_rev[2] = pathA[0];

          Vec<Int, 3> globalA =
              atom_local_to_global_indices(pathA_rev, rot_coord_offset1);
          Vec<Int, 3> globalB =
              atom_local_to_global_indices(pathB, rot_coord_offset2);

          int lenA = 0;
          for (int pi = 0; pi < 3; pi++)
            if (globalA[pi] != -1) lenA++;
          int lenB = 0;
          for (int pi = 0; pi < 3; pi++)
            if (globalB[pi] != -1) lenB++;
          if (lenA + lenB != 4) continue;

          Vec<Int, 4> atoms;
          atoms[0] = atoms[1] = atoms[2] = atoms[3] = -1;
          for (int pi = 0; pi < lenA; pi++) atoms[pi] = globalA[3 - lenA + pi];
          for (int pi = 0; pi < lenB; pi++) atoms[lenA + pi] = globalB[pi];

          Vec<Int, 4> local_indices;
          local_indices[0] = local_indices[1] = local_indices[2] =
              local_indices[3] = -1;
          for (int pi = 0; pi < lenA; pi++)
            local_indices[pi] = pathA_rev[3 - lenA + pi];
          for (int pi = 0; pi < lenB; pi++)
            local_indices[lenA + pi] = pathB[pi];

          Vec<Int, GB_MAX_HIER_DEPTH> h[4];
          for (int pos = 0; pos < 4; pos++) {
            int loc = local_indices[pos];
            int bt = (pos < lenA) ? block_type1 : block_type2;
            if (loc >= 0) {
              h[pos] = gen_atom_type_hierarchy[bt][loc];
            } else {
              h[pos] = Vec<Int, GB_MAX_HIER_DEPTH>::Constant(-1);
            }
          }

          int val_idx = inter_block_torsion_lookup<Int, D>(
              h[0],
              h[1],
              h[2],
              h[3],
              bond_type_int,
              gen_inter_torsion_hash_keys);
          if (val_idx >= 0) {
            score_torsion(atoms, gen_inter_torsion_hash_values[val_idx]);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_inter);
    }

    auto reduce_and_write = ([&] TMOL_DEVICE_FUNC(int tid) {
      Real const cta_score = DeviceOps<D>::template reduce_in_workgroup<nt>(
          torsion_score, shared, mgpu::plus_t<Real>());
      if (tid == 0 && cta_score != 0.0) {
        int const out_idx =
            output_block_pair_energies ? cta : dispatch_indices[0][cta];
        accumulate<D, Real>::add(V[0][out_idx], cta_score);
      }
    });
    DeviceOps<D>::template for_each_in_workgroup<nt>(reduce_and_write);
  });
  DeviceOps<D>::template foreach_workgroup<launch_t>(
      n_output_intxns_total, eval_torsions_for_interaction);

  return {
      V_t,
      dV_dx_t,
      dispatch_indices_t,
      n_output_intxns_for_rot_conn_offset_t,
      rotconn_for_output_intxn_t,
  };
}

// ===========================================================================
// GenBondedRotamerScoreDispatch::backward
// ===========================================================================
template <
    template <tmol::Device> class DeviceOps,
    tmol::Device D,
    typename Real,
    typename Int>
auto GenBondedRotamerScoreDispatch<DeviceOps, D, Real, Int>::backward(
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
    TView<Real, 2, D> dTdV) -> TPack<Vec<Real, 3>, 2, D> {
  int const n_atoms = rot_coords.size(0);
  int const n_block_types = gen_intra_subgraph_offsets.size(0);
  int const n_total_intra = gen_intra_subgraphs.size(0);
  int const max_n_conns = pose_stack_inter_block_connections.size(2);
  int const n_output_intxns_total = dispatch_indices.size(1);

  LAUNCH_BOX_32;
  CTA_REAL_REDUCE_T_TYPEDEF;

  auto dV_dx_t = TPack<Vec<Real, 3>, 2, D>::zeros({1, n_atoms});
  auto dV_dx = dV_dx_t.view;

  auto eval_torsions_for_interaction = ([=] TMOL_DEVICE_FUNC(int cta) {
    int const rotconn_ind = rotconn_for_output_intxn[cta];
    int const rot_ind1 = rotconn_ind / (max_n_conns + 1);
    int const conn_ind1 = rotconn_ind % (max_n_conns + 1);
    int const pose_ind = pose_ind_for_rot[rot_ind1];
    int const block_type1 = block_type_ind_for_rot[rot_ind1];
    int const rot_coord_offset1 = rot_coord_offset[rot_ind1];
    Real const block_weight = dTdV[0][cta];

    auto score_torsion =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          Real dummy = 0.0;
          auto eval = gbtorsion_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],
              prm[1],
              prm[2],
              prm[3],
              prm[4]);
          accumulate_torsion_result<Real, Int, D>(
              dummy, eval, atoms, true, dV_dx[0], block_weight);
        });

    auto score_improper =
        ([&] TMOL_DEVICE_FUNC(Vec<Int, 4> atoms, Vec<Real, 5> prm) {
          Real dummy = 0.0;
          auto eval = gbimproper_V_dV(
              rot_coords[atoms[0]],
              rot_coords[atoms[1]],
              rot_coords[atoms[2]],
              rot_coords[atoms[3]],
              prm[0],
              prm[1]);
          accumulate_torsion_result<Real, Int, D>(
              dummy, eval, atoms, true, dV_dx[0], block_weight);
        });

    if (conn_ind1 == max_n_conns) {
      int const intra_start = gen_intra_subgraph_offsets[block_type1];
      int const intra_end = (block_type1 + 1 == n_block_types)
                                ? n_total_intra
                                : gen_intra_subgraph_offsets[block_type1 + 1];
      int const n_intra = intra_end - intra_start;

      auto eval_intra = ([&] TMOL_DEVICE_FUNC(int tid) {
        for (int i = tid; i < n_intra; i += nt) {
          int const idx = intra_start + i;
          Vec<Int, 5> sg = gen_intra_subgraphs[idx];
          if (sg[0] == -1) continue;
          Vec<Int, 4> local_sg;
          local_sg[0] = sg[1];
          local_sg[1] = sg[2];
          local_sg[2] = sg[3];
          local_sg[3] = sg[4];
          Vec<Int, 4> atoms =
              atom_local_to_global_indices(local_sg, rot_coord_offset1);
          Vec<Real, 5> prm = gen_intra_params[idx];
          if (sg[0] == 0) {
            score_torsion(atoms, prm);
          } else {
            score_improper(atoms, prm);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_intra);

    } else {
      int const block_ind1 = block_ind_for_rot[rot_ind1];
      int const block_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [0];
      int const conn_ind2 =
          pose_stack_inter_block_connections[pose_ind][block_ind1][conn_ind1]
                                            [1];
      int const local_rot_ind2 =
          cta - n_output_intxns_for_rot_conn_offset[rotconn_ind];
      int const rot_ind2 =
          rot_offset_for_block[pose_ind][block_ind2] + local_rot_ind2;
      int const block_type2 = block_type_ind_for_rot[rot_ind2];
      int const rot_coord_offset2 = rot_coord_offset[rot_ind2];
      Int const bond_type_int =
          gen_connection_bond_types[block_type1][conn_ind1];

      auto eval_inter = ([&] TMOL_DEVICE_FUNC(int tid) {
        int const n_pairs = MAX_PATHS_FROM_CONN * MAX_PATHS_FROM_CONN;
        for (int pair = tid; pair < n_pairs; pair += nt) {
          int const path_A_idx = pair / MAX_PATHS_FROM_CONN;
          int const path_B_idx = pair % MAX_PATHS_FROM_CONN;
          Vec<Int, 3> pathA =
              atom_paths_from_conn[block_type1][conn_ind1][path_A_idx];
          Vec<Int, 3> pathB =
              atom_paths_from_conn[block_type2][conn_ind2][path_B_idx];
          if (pathA[0] == -1 || pathB[0] == -1) continue;

          Vec<Int, 3> pathA_rev;
          pathA_rev[0] = pathA[2];
          pathA_rev[1] = pathA[1];
          pathA_rev[2] = pathA[0];

          Vec<Int, 3> globalA =
              atom_local_to_global_indices(pathA_rev, rot_coord_offset1);
          Vec<Int, 3> globalB =
              atom_local_to_global_indices(pathB, rot_coord_offset2);

          int lenA = 0;
          for (int pi = 0; pi < 3; pi++)
            if (globalA[pi] != -1) lenA++;
          int lenB = 0;
          for (int pi = 0; pi < 3; pi++)
            if (globalB[pi] != -1) lenB++;
          if (lenA + lenB != 4) continue;

          Vec<Int, 4> atoms;
          atoms[0] = atoms[1] = atoms[2] = atoms[3] = -1;
          for (int pi = 0; pi < lenA; pi++) atoms[pi] = globalA[3 - lenA + pi];
          for (int pi = 0; pi < lenB; pi++) atoms[lenA + pi] = globalB[pi];

          Vec<Int, 4> local_indices;
          local_indices[0] = local_indices[1] = local_indices[2] =
              local_indices[3] = -1;
          for (int pi = 0; pi < lenA; pi++)
            local_indices[pi] = pathA_rev[3 - lenA + pi];
          for (int pi = 0; pi < lenB; pi++)
            local_indices[lenA + pi] = pathB[pi];

          Vec<Int, GB_MAX_HIER_DEPTH> h[4];
          for (int pos = 0; pos < 4; pos++) {
            int loc = local_indices[pos];
            int bt = (pos < lenA) ? block_type1 : block_type2;
            if (loc >= 0) {
              h[pos] = gen_atom_type_hierarchy[bt][loc];
            } else {
              h[pos] = Vec<Int, GB_MAX_HIER_DEPTH>::Constant(-1);
            }
          }

          int val_idx = inter_block_torsion_lookup<Int, D>(
              h[0],
              h[1],
              h[2],
              h[3],
              bond_type_int,
              gen_inter_torsion_hash_keys);
          if (val_idx >= 0) {
            score_torsion(atoms, gen_inter_torsion_hash_values[val_idx]);
          }
        }
      });
      DeviceOps<D>::template for_each_in_workgroup<nt>(eval_inter);
    }
  });
  DeviceOps<D>::template foreach_workgroup<launch_t>(
      n_output_intxns_total, eval_torsions_for_interaction);

  return dV_dx_t;
}

#undef Real3
#undef GB_MAX_HIER_DEPTH
#undef GB_BOND_WILDCARD

}  // namespace potentials
}  // namespace genbonded
}  // namespace score
}  // namespace tmol
