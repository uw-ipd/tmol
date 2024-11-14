#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include "common.hh"

namespace tmol {
namespace kinematics {

#ifdef __CUDACC__
#define gpuErrPeek gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__);
#define gpuErrSync gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__);
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
        stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#else
#define gpuErrPeek
#define gpuErrSync
#define gpuErrchk(ans) \
  { ans; }
#endif

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_kfo_indices_for_atoms(
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Int, 1, D> block_type_n_atoms,
    TView<bool, 2, D> block_type_atom_is_real)
    -> std::tuple<TPack<Int, 2, D>, TPack<Int, 2, D>, TPack<Int, 3, D>> {
  int const n_poses = pose_stack_block_coord_offset.size(0);
  int const max_n_blocks = pose_stack_block_coord_offset.size(1);
  int const max_n_atoms_per_block = block_type_atom_is_real.size(1);
  auto block_n_atoms_tp = TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto block_kfo_offset_tp = TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto block_n_atoms = block_n_atoms_tp.view;
  auto block_kfo_offset = block_kfo_offset_tp.view;

  LAUNCH_BOX_32;

  // 1. Look up n atoms per block, adding one for the root to block[0][0]
  // 2. Scan to get offsets
  // 3. Read back n-kfo-atoms total???
  // 4. Write down KFO index for each real atom

  auto get_n_atoms_for_block = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose = ind / max_n_blocks;
    int const block = ind % max_n_blocks;
    int const block_type = pose_stack_block_type[pose][block];

    // add in an extra atom for the root!
    int const root_offset = (pose == 0 && block == 0) ? 1 : 0;
    int n_block_atoms = 0;
    if (block_type != -1) {
      n_block_atoms = block_type_n_atoms[block_type];
    }
    block_n_atoms[pose][block] = n_block_atoms + root_offset;
  });

  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, get_n_atoms_for_block);
  Int n_kfo_atoms =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          block_n_atoms.data(),
          block_kfo_offset.data(),
          n_poses * max_n_blocks,
          mgpu::plus_t<Int>());

  auto kfo_2_orig_mapping_tp = TPack<Int, 2, D>::full({n_kfo_atoms, 3}, -1);
  auto atom_kfo_index_tp = TPack<Int, 3, D>::full(
      {n_poses, max_n_blocks, max_n_atoms_per_block}, -1);
  auto kfo_2_orig_mapping = kfo_2_orig_mapping_tp.view;
  auto atom_kfo_index = atom_kfo_index_tp.view;

  auto get_kfo_mapping = ([=] TMOL_DEVICE_FUNC(int ind) {
    int const pose = ind / (max_n_blocks * max_n_atoms_per_block);
    ind = ind - pose * (max_n_blocks * max_n_atoms_per_block);
    int const block = ind / max_n_atoms_per_block;
    int const atom = ind % max_n_atoms_per_block;
    int const block_type = pose_stack_block_type[pose][block];

    int kfo_offset = block_kfo_offset[pose][block];

    if (pose == 0 && block == 0) {
      kfo_offset = 1;
      if (atom == 0) {
        block_kfo_offset[pose][block] = 1;
      }
    }
    if (block_type != -1) {
      // correct [0, 0]
      bool atom_is_real = block_type_atom_is_real[block_type][atom];
      if (atom_is_real) {
        int kfo_ind = kfo_offset + atom;
        atom_kfo_index[pose][block][atom] = kfo_ind;
        kfo_2_orig_mapping[kfo_ind][0] = pose;
        kfo_2_orig_mapping[kfo_ind][1] = block;
        kfo_2_orig_mapping[kfo_ind][2] = atom;
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks * max_n_atoms_per_block, get_kfo_mapping);

  return std::make_tuple(
      block_kfo_offset_tp, kfo_2_orig_mapping_tp, atom_kfo_index_tp);
}

// P -- number of Poses
// L -- length of the longest Pose
// C -- the maximum number of inter-residue connections
// T -- number of block types
// O -- number of output connection types; i.e. max-n-conn + 1
// A -- maximum number of atoms in a block
// C = maximum number of inter-residue connections in any block type
// E = maximum number of edges in any one FoldTree of the FoldForest
// I = maximum number of input connections in any block type
// O = maximum number of output connections in any block type
// G = maximum number of generations in any block type
// N = maximum number of nodes in any generation in any block type
// S = maximum number of scan path segments in any generation in any block type
template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::
    get_block_parent_connectivity_from_toposort(
        TView<Int, 2, D> pose_stack_block_type,                 // P x L
        TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
        TView<Int, 2, D> pose_stack_ff_parent,
        TView<Int, 2, D> dfs_order_of_ff_edges,
        TView<Int, 1, D> n_ff_edges,               // P
        TView<Int, 3, D> ff_edges,                 // P x E x 4
        TView<Int, 2, D> first_ff_edge_for_block,  // P x L
        // TView<Int, 2, D> max_n_gens_for_ff_edge, // P x E
        TView<Int, 2, D> first_child_of_ff_edge,    // P x E
        TView<Int, 2, D> delay_for_edge,            // P x E
        TView<Int, 1, D> topo_sort_index_for_edge,  // (P*E)
        TView<Int, 1, D> block_type_n_conn,         // T
        TView<Int, 2, D>
            block_type_polymeric_conn_index  // T x 2 - 2 is for "down" and "up"
                                             // connections.

        ) -> TPack<Int, 3, D> {
  using namespace tmol::score::common;
  LAUNCH_BOX_32;
  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_ff_edges_per_pose = ff_edges.size(1);

  auto pose_stack_block_in_and_first_out_t =
      TPack<Int, 3, D>::full({n_poses, max_n_blocks, 2}, -1);
  auto pose_stack_block_in_and_first_out =
      pose_stack_block_in_and_first_out_t.view;

  // 1. Get the parent block of each block
  // printf("get_block_parent_connectivity_from_toposort 1\n");
  auto get_parent_connections = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / max_n_blocks;
    int const block = i % max_n_blocks;
    int const block_type = pose_stack_block_type[pose][block];
    if (block_type == -1) {
      return;
    }
    // printf(
    //     "get_parent_connections p: %d b: %d bt: %d\n", pose, block,
    //     block_type);
    int const ff_edge = first_ff_edge_for_block[pose][block];
    // printf("ff_edge %d\n", ff_edge);
    int const edge_type = ff_edges[pose][ff_edge][0];
    int const parent_block = pose_stack_ff_parent[pose][block];
    // printf(
    //     "ff_edge %d edge_type %d parent_block %d\n",
    //     ff_edge,
    //     edge_type,
    //     parent_block);
    if (parent_block != -1) {
      int const parent_ff_edge = first_ff_edge_for_block[pose][parent_block];
      if (ff_edge == parent_ff_edge) {
        // parent is in the same FF edge
        if (edge_type == 0) {
          // currently only support polymer (peptide) edges and jumps; no
          // "chemical" edges just yet
          int const parent_block_type =
              pose_stack_block_type[pose][parent_block];
          int const conn_to_parent =
              block_type_polymeric_conn_index[block_type]
                                             [(parent_block < block) ? 0 : 1];
          int const conn_to_child =
              block_type_polymeric_conn_index[parent_block_type]
                                             [(parent_block < block) ? 1 : 0];
          pose_stack_block_in_and_first_out[pose][block][0] = conn_to_parent;
          pose_stack_block_in_and_first_out[pose][parent_block][1] =
              conn_to_child;
        } else {
          // The "first edge" for the root block may in fact be a jump
          // printf(
          //     "block in for jump edge %d %d (%d): %d\n",
          //     pose,
          //     block,
          //     block_type,
          //     block_type_n_conn[block_type]);
          pose_stack_block_in_and_first_out[pose][block][0] =
              block_type_n_conn[block_type];
        }
      } else {
        if (edge_type == 0) {
          // polymer edge
          int conn_to_parent =
              block_type_polymeric_conn_index[block_type]
                                             [(parent_block < block) ? 0 : 1];
          pose_stack_block_in_and_first_out[pose][block][0] = conn_to_parent;

        } else {
          // jump edge
          // assert edge_type == 1
          // printf(
          //     "block in for jump edge %d %d (%d): %d\n",
          //     pose,
          //     block,
          //     block_type,
          //     block_type_n_conn[block_type]);
          pose_stack_block_in_and_first_out[pose][block][0] =
              block_type_n_conn[block_type];
        }
      }
    } else {
      // printf("looking at the root block, ff_edge %d\n", ff_edge);
      // looking at the root block
      // "root connection" index is n_conn + 1
      pose_stack_block_in_and_first_out[pose][block][0] =
          block_type_n_conn[block_type] + 1;
      // int const edge_first_child = first_child_of_ff_edge[pose][ff_edge];
      int const edge_type = ff_edges[pose][ff_edge][0];
      int const end_block = ff_edges[pose][ff_edge][2];
      if (edge_type == 0) {
        // polymer edge
        int conn_toward_end =
            block_type_polymeric_conn_index[block_type]
                                           [(block < end_block) ? 1 : 0];
        pose_stack_block_in_and_first_out[pose][block][1] = conn_toward_end;
      } else {
        // jump edge
        // assert edge_type == 1
        pose_stack_block_in_and_first_out[pose][block][1] =
            block_type_n_conn[block_type];
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, get_parent_connections);

  // printf("get_block_parent_connectivity_from_toposort 2\n");
  // Also handle the first output connection for the end residue of each edge
  auto set_output_conn_for_edge_end = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / max_n_ff_edges_per_pose;
    int const edge = i % max_n_ff_edges_per_pose;
    int const edge_type = ff_edges[pose][edge][0];
    if (edge_type == -1) {
      return;
    }

    // int const edge_start_block = ff_edges[pose][edge][1];
    int const edge_end_block = ff_edges[pose][edge][2];
    int const block_type = pose_stack_block_type[pose][edge_end_block];
    int const edge_first_child = first_child_of_ff_edge[pose][edge];
    if (edge_first_child != -1) {
      int const first_child_edge_type = ff_edges[pose][edge_first_child][0];
      if (first_child_edge_type == 0) {
        // polymer edge
        int const first_child_end_block = ff_edges[pose][edge_first_child][2];
        // int const block_type = pose_stack_block_type[pose][edge_end_block];
        pose_stack_block_in_and_first_out[pose][edge_end_block][1] =
            block_type_polymeric_conn_index
                [block_type][(edge_end_block < first_child_end_block) ? 1 : 0];
      } else {
        printf(
            "pose %d edge %d end block %d edge type %d\n",
            pose,
            edge,
            edge_end_block,
            edge_type);
        // jump edge
        // assert edge_type == 1
        // jump connection denoted by n_conn.
        pose_stack_block_in_and_first_out[pose][edge_end_block][1] =
            block_type_n_conn[block_type];
      }
    } else {
      // leaf nodes: these are denoted with an output connection of n_conn + 1
      int const n_conn = block_type_n_conn[block_type];
      pose_stack_block_in_and_first_out[pose][edge_end_block][1] = n_conn + 1;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_ff_edges_per_pose, set_output_conn_for_edge_end);

  // TEMP!
  // for (int pose = 0; pose < n_poses; ++pose) {
  //   for (int block = 0; block < max_n_blocks; ++block) {
  //     printf(
  //         "pose_stack_block_in_and_first_out[%d][%d][:] %d %d\n",
  //         pose,
  //         block,
  //         pose_stack_block_in_and_first_out[pose][block][0],
  //         pose_stack_block_in_and_first_out[pose][block][1]);
  //   }
  // }

  // printf("get_block_parent_connectivity_from_toposort done\n");
  return pose_stack_block_in_and_first_out_t;
}

// P -- number of Poses
// L -- length of the longest Pose
// C -- the maximum number of inter-residue connections
// T -- number of block types
// O -- number of output connection types; i.e. max-n-conn + 1 (TO DO??
// max-n-conn + 2???) A -- maximum number of atoms in a block

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_kfo_atom_parents(
    TView<Int, 2, D> pose_stack_block_type,                 // P x L
    TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
    TView<Int, 2, D> pose_stack_ff_parent,                  // P x L
    // TView<Int, 2, D> pose_stack_ff_conn_to_parent,          // P x L --
    // redundant
    TView<Int, 3, D> pose_stack_block_in_and_first_out,  // P x L x 2
    TView<Int, 3, D> block_type_parents,                 // T x O x A
    TView<Int, 2, D> kfo_2_orig_mapping,                 // K x 3
    TView<Int, 3, D> atom_kfo_index,                     // P x L x A
    TView<Int, 1, D> block_type_jump_atom,               // T
    TView<Int, 1, D> block_type_n_conn,                  // T
    TView<Int, 2, D> block_type_conn_atom                // T x C
    ) -> std::tuple<TPack<Int, 1, D>, TPack<Int, 1, D>> {
  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_atoms_per_block = block_type_parents.size(2);
  int const n_kfo_atoms = kfo_2_orig_mapping.size(0);

  auto block_n_atoms_tp = TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto block_kfo_offset_tp = TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto block_n_atoms = block_n_atoms_tp.view;
  auto block_kfo_offset = block_kfo_offset_tp.view;

  LAUNCH_BOX_32;

  auto kfo_parent_atoms_t = TPack<Int, 1, D>::zeros({n_kfo_atoms});
  auto kfo_grandparent_atoms_t = TPack<Int, 1, D>::zeros({n_kfo_atoms});
  auto kfo_parent_atoms = kfo_parent_atoms_t.view;
  auto kfo_grandparent_atoms = kfo_grandparent_atoms_t.view;

  auto get_parent_atoms = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = kfo_2_orig_mapping[i][0];
    int const block = kfo_2_orig_mapping[i][1];
    int const atom = kfo_2_orig_mapping[i][2];
    if (pose == -1) {
      return;
    }
    int const block_type = pose_stack_block_type[pose][block];
    int const conn_to_parent =
        pose_stack_block_in_and_first_out[pose][block][0];
    // pose_stack_ff_conn_to_parent[pose][block];
    // int const ff_in = ;

    int const bt_parent_for_atom =
        block_type_parents[block_type][conn_to_parent][atom];
    // printf(
    //     "pose %d block %d atom %d block_type %d conn_to_parent %d "
    //     "bt_parent_for_atom %d\n",
    //     pose,
    //     block,
    //     atom,
    //     block_type,
    //     conn_to_parent,
    //     bt_parent_for_atom);
    if (bt_parent_for_atom < 0) {
      // Inter-residue connection
      int const parent_block = pose_stack_ff_parent[pose][block];
      // printf("parent_block %d\n", parent_block);
      if (parent_block == -1) {
        // Root connection -- the root is at 0
        kfo_parent_atoms[i] = 0;
      } else {
        int const n_conn = block_type_n_conn[block_type];
        if (conn_to_parent == n_conn) {
          // Jump connection
          int const parent_block_type =
              pose_stack_block_type[pose][parent_block];
          int const jump_atom = block_type_jump_atom[parent_block_type];
          kfo_parent_atoms[i] = atom_kfo_index[pose][parent_block][jump_atom];
        } else {
          // Use inter-block connectivity info from PoseStack
          int const parent_block_type =
              pose_stack_block_type[pose][parent_block];
          // printf("parent_block_type %d\n", parent_block_type);
          int const parent_conn =
              pose_stack_inter_residue_connections[pose][block][conn_to_parent]
                                                  [1];
          // printf("parent_conn %d\n", parent_conn);
          int const parent_conn_atom =
              block_type_conn_atom[parent_block_type][parent_conn];
          // printf("parent_conn_atom %d\n", parent_conn_atom);
          kfo_parent_atoms[i] =
              atom_kfo_index[pose][parent_block][parent_conn_atom];
        }
      }
    } else {
      // Intra-residue parent
      kfo_parent_atoms[i] = atom_kfo_index[pose][block][bt_parent_for_atom];
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(n_kfo_atoms, get_parent_atoms);

  // second step: look up parent's parent. All atoms have a parent, even the
  // root which is its own parent.
  auto get_grandparent_atoms = ([=] TMOL_DEVICE_FUNC(int i) {
    int const parent = kfo_parent_atoms[i];
    kfo_grandparent_atoms[i] = kfo_parent_atoms[parent];
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_kfo_atoms, get_grandparent_atoms);
  return {kfo_parent_atoms_t, kfo_grandparent_atoms_t};
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_children(
    TView<Int, 2, D> pose_stack_block_type,  // x
    TView<Int, 3, D>
        pose_stack_block_in_and_first_out,  // x pose_stack_ff_conn_to_parent
    TView<Int, 2, D> kfo_2_orig_mapping,    // x
    TView<Int, 1, D> kfo_parent_atoms,      // x
    TView<Int, 1, D> block_type_n_conn      // x
    )
    -> std::tuple<
        TPack<Int, 1, D>,
        TPack<Int, 1, D>,
        TPack<Int, 1, D>,
        TPack<bool, 1, D>> {
  using namespace tmol::score::common;
  int const n_kfo_atoms = kfo_2_orig_mapping.size(0);

  LAUNCH_BOX_32;

  // Now let's go and assign child-atom lists for each atom
  auto child_list_t = TPack<Int, 1, D>::full({n_kfo_atoms}, -1);
  auto child_list_span_t = TPack<Int, 1, D>::zeros({n_kfo_atoms + 1});
  auto n_children_t = TPack<Int, 1, D>::zeros(
      {n_kfo_atoms + 1});  // leave one extra space for scan
  auto n_jump_children_t = TPack<Int, 1, D>::zeros({n_kfo_atoms + 1});
  auto n_non_jump_children_t = TPack<Int, 1, D>::zeros({n_kfo_atoms + 1});
  auto count_n_non_jump_children_t = TPack<Int, 1, D>::zeros({n_kfo_atoms});
  auto count_jump_children_t = TPack<Int, 1, D>::zeros({n_kfo_atoms});
  auto is_atom_jump_t = TPack<bool, 1, D>::zeros({n_kfo_atoms});

  auto child_list = child_list_t.view;
  auto child_list_span = child_list_span_t.view;
  auto n_children = n_children_t.view;
  auto n_jump_children = n_jump_children_t.view;
  auto n_non_jump_children = n_non_jump_children_t.view;
  auto count_n_non_jump_children = count_n_non_jump_children_t.view;
  auto count_jump_children = count_jump_children_t.view;
  auto is_atom_jump = is_atom_jump_t.view;

  auto count_children_for_parent = ([=] TMOL_DEVICE_FUNC(int i) {
    // Each atom looks up its parent and atomic-increments its parent's
    // child count; either recording that it's a jump child or that
    // it's a non-jump child.
    // As a knock-on, it also records whether it is a jump atom.
    int const parent = kfo_parent_atoms[i];
    if (parent == i) {
      // nothing to be done for the root; also, it doesn't have a valid
      // entry in the Pose, so, subseqent lookups would fail.
      return;
    }
    int const pose = kfo_2_orig_mapping[i][0];
    int const block = kfo_2_orig_mapping[i][1];
    int const atom = kfo_2_orig_mapping[i][2];
    int const block_type = pose_stack_block_type[pose][block];
    // printf("count_children_for_parent %d %d %d %d %d\n", i, pose, block,
    // atom, parent);
    if (parent == 0) {
      // This atom's parent is the root and is connected to it by a jump
      Int one(1);
      accumulate<D, Int>::add(n_jump_children[parent], one);
      is_atom_jump[i] = true;
    } else {
      int const parent_block = kfo_2_orig_mapping[parent][1];
      // printf("parent_block %d\n", parent_block);
      if (parent_block == block) {
        // Intra-residue connection
        accumulate<D, Int>::add(n_non_jump_children[parent], 1);
      } else {
        // Inter-residue connection, but, is it a jump connetion?
        int const n_conn = block_type_n_conn[block_type];
        int const conn_to_parent =
            pose_stack_block_in_and_first_out[pose][block][0];
        // printf("n_conn %d conn_to_parent %d\n", n_conn, conn_to_parent);
        if (conn_to_parent >= n_conn) {
          // Jump connection
          accumulate<D, Int>::add(n_jump_children[parent], 1);
          is_atom_jump[i] = true;
        } else {
          // Non-jump connection
          accumulate<D, Int>::add(n_non_jump_children[parent], 1);
        }
      }
    }
  });
  // printf("count_children_for_parent %d\n", n_kfo_atoms);
  DeviceDispatch<D>::template forall<launch_t>(
      n_kfo_atoms, count_children_for_parent);

  auto sum_jump_and_non_jump_children = ([=] TMOL_DEVICE_FUNC(int i) {
    // Now each atom looks at how many jump and non-jump children it has.
    n_children[i] = n_non_jump_children[i] + n_jump_children[i];
  });
  // printf("sum_jump_and_non_jump_children %d\n", n_kfo_atoms);
  DeviceDispatch<D>::template forall<launch_t>(
      n_kfo_atoms, sum_jump_and_non_jump_children);

  // Now get the beginning and end indices for the child-list ranges.
  // printf("scan n_children %d\n", n_kfo_atoms);
  DeviceDispatch<D>::template scan<mgpu::scan_type_exc>(
      n_children.data(),
      child_list_span.data(),
      n_kfo_atoms + 1,
      mgpu::plus_t<Int>());

  // Okay, now ask each atom to insert itself into its parent's child-list
  auto fill_child_list = ([=] TMOL_DEVICE_FUNC(int i) {
    int const parent = kfo_parent_atoms[i];
    if (parent == i) {
      // nothing to be done for the root
      return;
    }
    bool is_jump = is_atom_jump[i];
    if (is_jump) {
      int const jump_offset =
          accumulate<D, Int>::add(count_jump_children[parent], 1);
      int const jump_start = child_list_span[parent];
      child_list[jump_start + jump_offset] = i;
      // printf("fill_child_list jump %d %d %d %d %d %d\n", i, parent,
      // jump_offset, jump_start, child_list_span[parent],
      // n_jump_children[parent]);
    } else {
      int const non_jump_offset =
          accumulate<D, Int>::add(count_n_non_jump_children[parent], 1);
      int const non_jump_start =
          child_list_span[parent] + n_jump_children[parent];
      child_list[non_jump_start + non_jump_offset] = i;
      // printf("fill_child_list non-jump %d %d %d %d %d %d\n", i, parent,
      // non_jump_offset, non_jump_start, child_list_span[parent],
      // n_jump_children[parent]);
    }
  });
  // printf("fill_child_list %d\n", n_kfo_atoms);
  DeviceDispatch<D>::template forall<launch_t>(n_kfo_atoms, fill_child_list);

  // TO DO: replace with segmented sort!

  // Finally, we need to sort the child lists by atom index because
  // the fill_child_list operation is not deterministic on the GPU
  // and we want to ensure that the child-lists are deterministic
  // because they will determine the connectivity of the KinForest.
  // By having each atom sort its own children, we avoid any race
  // conditions.
  auto sort_child_list = ([=] TMOL_DEVICE_FUNC(int i) {
    int const start = child_list_span[i];
    int const end = child_list_span[i + 1];
    if (end - start > 1) {
      // The jump atoms must come first, then the non-jump atoms
      int const n_my_jump_children = n_jump_children[i];
      // bubble sort
      for (int j = 0; j < n_my_jump_children; ++j) {
        for (int k = 0; k < n_my_jump_children - j - 1; ++k) {
          int const a = child_list[start + k];
          int const b = child_list[start + k + 1];
          // printf("bubble sort jump children %d, %d %d, %d: %d %d %d %d\n",
          //    i, start, end, n_my_jump_children, j, k, a, b);
          if (a > b) {
            child_list[start + k] = b;
            child_list[start + k + 1] = a;
          }
        }
      }
      for (int j = 0; j < end - start; ++j) {
        for (int k = 0; k < end - start - j - 1; ++k) {
          int const a = child_list[start + k];
          int const b = child_list[start + k + 1];
          // printf("bubble sort non-jump children %d, %d %d, %d: %d %d %d
          // %d\n",
          //    i, start, end, n_my_jump_children, j, k, a, b);
          if (a > b) {
            child_list[start + k] = b;
            child_list[start + k + 1] = a;
          }
        }
      }
    }
  });
  // printf("sort_child_list %d\n", n_kfo_atoms);
  DeviceDispatch<D>::template forall<launch_t>(n_kfo_atoms, sort_child_list);
  return {n_children_t, child_list_span_t, child_list_t, is_atom_jump_t};
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_id_and_frame_xyz(
    int64_t const max_n_pose_atoms,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> kfo_2_orig_mapping,  // K x 3
    TView<Int, 1, D> parents,             // K
    TView<Int, 1, D> child_list_span,     // K+1
    TView<Int, 1, D> child_list,          // K
    TView<bool, 1, D> is_atom_jump        // K
    )
    -> std::tuple<
        TPack<Int, 1, D>,
        TPack<Int, 1, D>,
        TPack<Int, 1, D>,
        TPack<Int, 1, D>> {
  LAUNCH_BOX_32;
  int const n_kintree_nodes = parents.size(0);

  auto id_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto frame_x_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto frame_y_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto frame_z_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto id = id_t.view;
  auto frame_x = frame_x_t.view;
  auto frame_y = frame_y_t.view;
  auto frame_z = frame_z_t.view;

  auto first_pass_frame_xyz = ([=] TMOL_DEVICE_FUNC(int i) {
    if (i == 0) {
      id[i] = -1;
    } else {
      int const pose = kfo_2_orig_mapping[i][0];
      int const block = kfo_2_orig_mapping[i][1];
      int const atom = kfo_2_orig_mapping[i][2];
      // ID represents the position of the atom in a flattened
      // version of the pose-stack coords tensor
      id[i] = pose * max_n_pose_atoms
              + pose_stack_block_coord_offset[pose][block] + atom;
    }
    frame_x[i] = i;
    int parent = parents[i];
    // printf("first_pass_frame_xyz %d %d\n", i, parent);
    frame_y[i] = parent;
    frame_z[i] = parents[parent];
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_kintree_nodes, first_pass_frame_xyz);

  auto stub_defined_for_jump_atom = ([=] TMOL_DEVICE_FUNC(int jump_atom) {
    int first_nonjump_child = -1;
    for (int child_ind = child_list_span[jump_atom];
         child_ind < child_list_span[jump_atom + 1];
         ++child_ind) {
      int child_atom = child_list[child_ind];
      if (is_atom_jump[child_atom]) {
        continue;
      }
      if (first_nonjump_child == -1) {
        first_nonjump_child = child_atom;
      } else {
        return true;
      }
    }
    if (first_nonjump_child != -1) {
      for (int grandchild_ind = child_list_span[first_nonjump_child];
           grandchild_ind < child_list_span[first_nonjump_child + 1];
           ++grandchild_ind) {
        if (!is_atom_jump[child_list[grandchild_ind]]) {
          return true;
        }
      }
    }
    return false;
  });

  // "Recursive" function for finding an acceptible set of
  // child1 and child2 atoms for a jump atom. Handles cases when
  // there are too few children, or when the children are
  // themselves jumps.
  auto get_c1_and_c2_atoms = ([=] TMOL_DEVICE_FUNC(int jump_atom) {
    while (true) {
      int first_nonjump_child = -1;
      int second_nonjump_child = -1;
      for (int child_ind = child_list_span[jump_atom];
           child_ind < child_list_span[jump_atom + 1];
           ++child_ind) {
        int child_atom = child_list[child_ind];
        if (is_atom_jump[child_atom]) {
          continue;
        }
        if (first_nonjump_child == -1) {
          first_nonjump_child = child_atom;
        } else {
          second_nonjump_child = child_atom;
          break;
        }
      }
      if (first_nonjump_child == -1) {
        // No non-jump children. "Recurse" to parent.
        int jump_parent = parents[jump_atom];
        assert(jump_parent != jump_atom);
        jump_atom = jump_parent;
        continue;
      }
      for (int grandchild_ind = child_list_span[first_nonjump_child];
           grandchild_ind < child_list_span[first_nonjump_child + 1];
           ++grandchild_ind) {
        int grandchild_atom = child_list[grandchild_ind];
        if (!is_atom_jump[grandchild_atom]) {
          // printf(
          //     "get_c1_and_c2_atoms: jump atom %d, %d, %d\n",
          //     jump_atom,
          //     first_nonjump_child,
          //     grandchild_atom);
          return std::make_tuple(first_nonjump_child, grandchild_atom);
        }
      }
      if (second_nonjump_child == -1) {
        // Insufficient non-jump descendants. "Recurse" to parent
        int jump_parent = parents[jump_atom];
        assert(jump_parent != jump_atom);
        jump_atom = jump_parent;
        continue;
      }
      // printf(
      //     "get_c1_and_c2_atoms: jump atom %d, %d, %d\n",
      //     jump_atom,
      //     first_nonjump_child,
      //     second_nonjump_child);
      return std::make_tuple(first_nonjump_child, second_nonjump_child);
    }
  });

  auto fix_jump_node = ([=] TMOL_DEVICE_FUNC(int i) {
    int c1 = 0;
    int c2 = 0;
    if (is_atom_jump[i]) {
      bool is_root = parents[i] == 0;
      if (is_root) {
        auto result = get_c1_and_c2_atoms(i);
        c1 = std::get<0>(result);
        c2 = std::get<1>(result);
        // printf("c1 c2 %d %d\n", c1, c2);

        frame_x[i] = c1;
        frame_y[i] = i;
        frame_z[i] = c2;

        frame_x[c1] = c1;
        frame_y[c1] = i;
        frame_z[c1] = c2;

        for (int j = child_list_span[i] + 1; j < child_list_span[i + 1]; ++j) {
          int child = child_list[j];
          if (is_atom_jump[child]) {
            continue;
          }
          if (child == c1) {
            continue;
          }
          frame_x[child] = child;
          frame_y[child] = i;
          frame_z[child] = c1;
        }

      } else {
        if (stub_defined_for_jump_atom(i)) {
          auto result = get_c1_and_c2_atoms(i);
          c1 = std::get<0>(result);
          c2 = std::get<1>(result);
          // printf("c1 c2 %d %d\n", c1, c2);

          frame_x[i] = c1;
          frame_y[i] = i;
          frame_z[i] = c2;

          frame_x[c1] = c1;
          frame_y[c1] = i;
          frame_z[c1] = c2;

          for (int j = child_list_span[i] + 1; j < child_list_span[i + 1];
               ++j) {
            int child = child_list[j];
            if (is_atom_jump[child]) {
              continue;
            }
            if (child == c1) {
              continue;
            }
            frame_x[child] = child;
            frame_y[child] = i;
            frame_z[child] = c1;
          }
        } else {
          int parent = parents[i];
          auto result = get_c1_and_c2_atoms(parent);
          c1 = std::get<0>(result);
          c2 = std::get<1>(result);

          frame_x[i] = c1;
          frame_y[i] = i;
          frame_z[i] = c2;

          // The jump may have 1 non-jump child. It's not clear
          // what frame the child should have.
          for (int j = child_list_span[i]; j < child_list_span[i + 1]; ++j) {
            int child = child_list[j];
            if (is_atom_jump[child]) {
              continue;
            }
            frame_x[child] = c1;
            frame_y[child] = i;
            frame_z[child] = c2;
          }
        }
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(n_kintree_nodes, fix_jump_node);
  return {id_t, frame_x_t, frame_y_t, frame_z_t};
}

// P = number of poses
// L = length of the longest pose
// T = number of block types
// A = maximum number of atoms in any block type
// C = maximum number of inter-residue connections in any block type
// E = maximum number of edges in any one FoldTree of the FoldForest
// I = maximum number of input connections in any block type
// O = maximum number of output connections in any block type
// G = maximum number of generations in any block type
// N = maximum number of nodes in any generation in any block type
// S = maximum number of scan path segs in any generation in any block type
template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::calculate_ff_edge_delays(
    TView<Int, 2, D> pose_stack_block_coord_offset,  // P x L
    TView<Int, 2, D> pose_stack_block_type,          // x - P x L
    TView<Int, 3, Device::CPU> ff_edges_cpu,    // y - P x E x 4 -- 0: type, 1:
                                                // start, 2: stop, 3: jump ind
    TView<Int, 5, D> block_type_kts_conn_info,  // y - T x I x O x C x 2 -- 2 is
                                                // for gen (0) and scan (1)
    TView<Int, 5, D> block_type_nodes_for_gens,       // y - T x I x O x G x N
    TView<Int, 5, D> block_type_scan_path_seg_starts  // y - T x I x O x G x S
    )
    -> std::tuple<
        TPack<Int, 2, Device::CPU>,  // dfs_order_of_ff_edges_t
        TPack<Int, 1, Device::CPU>,  // n_ff_edges_t
        TPack<Int, 2, Device::CPU>,  // ff_edge_parent_t
        TPack<Int, 2, Device::CPU>,  // first_ff_edge_for_block_cpu_t
        TPack<Int, 2, Device::CPU>,  // pose_stack_ff_parent_t
        TPack<Int, 2, Device::CPU>,  // max_gen_depth_of_ff_edge_t
        TPack<Int, 2, Device::CPU>,  // first_child_of_ff_edge_t
        TPack<Int, 2, Device::CPU>,  // delay_for_edge_t
        TPack<Int, 1, Device::CPU>   // toposort_order_of_edges_t
        > {
  // The final step is to construct the nodes, scans, and gens tensors
  // from the per-block-type stencils.
  //

  // For each block, we need to know which FoldForest edge builds it.
  // For each FF edge, we need to know its generational delay.
  // With that, we can calculate the generational delay for each block.
  // For each scan-path segment, we need to know its offset into the nodes
  // tensor. For each scan-path segment, we need to know its offset into the
  // scan-path segment list. Then we can ask each scan-path segment how many
  // nodes it has, and generate the offsets using scan. We need to know how many
  // scan-path segments there are. We need to map scan-path segment index to
  // block, generation, and scan-path-segment-within-the-generation.

  // In order to know the index for any scan-path segment, we have
  // to count the number of block-scan paths that come before it. This can be
  // tricky because some block-scan paths continue into other blocks, and we do
  // not know a priori how many block-scan paths there are downstream of such a
  // block-scan path. For each (inter-block) scan path, we have to calculate how
  // many block-scan paths comprise it. Each scan path can be readily identified
  // from the fold forest. Each block type should identify which scan paths are
  // inter-block so it's easy to figure out for each block-scan path extend into
  // other blocks: not all do.

  // Step N-5:

  // Step N-4: count the number of blocks that build each (perhaps-multi-res)
  // scan path.

  // Step N-3: perform a segmented scan on the number of blocks that build each
  // (perhaps-multi-res) scan path.

  // Step N-2: write the number of atoms in each scan path to the appropriate
  // place in the n_atoms_for_scan_path_for_gen tensor.

  // Step N-1: perform a scan on the number of atoms in each scan path to get
  // the nodes tensor offset.

  // Step N: copy the scan path stencils into the nodes tensor, adding the
  // pose-stack- and block- offsets to the atom indices. Note that the upstream
  // jump atom must be added for jump edges that are the roots of paths.

  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_edges_per_ff = ff_edges_cpu.size(1);
  int const max_n_input_conn = block_type_kts_conn_info.size(1);
  int const max_n_output_conn = block_type_kts_conn_info.size(1);
  int const max_n_gens_per_bt = block_type_nodes_for_gens.size(3);
  int const max_n_nodes_per_gen = block_type_nodes_for_gens.size(4);
  int const max_n_scan_path_segs_per_gen =
      block_type_scan_path_seg_starts.size(4);

  // printf("n_poses %d max_n_edges_per_ff %d max_n_blocks %d\n", n_poses,
  // max_n_edges_per_ff, max_n_blocks);

  // Step 1:
  // printf("Step 1\n");
  // Construct a depth-first traversal of the fold-forest edges to determine a
  // partial order (and incidental total order) of the edges in the fold forest.
  // Do this by inserting all edges into an edge-list representation and then
  // starting at the root.
  auto dfs_order_of_ff_edges_t =
      TPack<Int, 2, Device::CPU>::zeros({n_poses, max_n_edges_per_ff});
  auto dfs_order_of_ff_edges = dfs_order_of_ff_edges_t.view;

  // ff_edge_parent is the index of the ff edge that is a parent of
  // the given edge.
  auto ff_edge_parent_t =
      TPack<Int, 2, Device::CPU>::zeros({n_poses, max_n_edges_per_ff});
  auto ff_edge_parent = ff_edge_parent_t.view;

  auto n_ff_edges_t =
      TPack<Int, 1, Device::CPU>::full({n_poses}, max_n_edges_per_ff);
  auto n_ff_edges = n_ff_edges_t.view;
  // auto block_has_children_t = TPack<bool, 2, Device::CPU>::zeros(
  //     {n_poses, max_n_res_per_pose});
  // auto block_has_children = block_has_children_t.view;

  std::vector<std::vector<std::list<std::tuple<int, int>>>> ff_children(
      n_poses);
  std::vector<std::vector<bool>> has_parent(n_poses);
  std::vector<std::vector<int>> edge_parent_for_block(n_poses);
  for (int pose = 0; pose < n_poses; ++pose) {
    ff_children[pose].resize(max_n_blocks);
    has_parent[pose].resize(max_n_blocks, false);
    edge_parent_for_block[pose].resize(max_n_blocks, -1);
    // for (int block = 0; block < max_n_blocks; ++block) {
    //   printf("initial set size: ff_children[%d][%d] %d\n", pose, block,
    //   ff_children[pose][block].size());
    // }
  }
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int edge = 0; edge < max_n_edges_per_ff; ++edge) {
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      // printf("ff_edge_type %d %d %d\n", pose, edge, ff_edge_type);
      if (ff_edge_type == -1) {
        n_ff_edges[pose] =
            edge;  // we are one past the last edge, thus at the number of edges
        continue;
      }
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      int const ff_edge_end = ff_edges_cpu[pose][edge][2];
      // printf(
      //     "%d %d %d %d %d\n",
      //     pose,
      //     edge,
      //     ff_edge_type,
      //     ff_edge_start,
      //     ff_edge_end);
      has_parent[pose][ff_edge_end] = true;
      // block_has_children[pose][ff_edge_start] = true;
      // The edge that ends at a given block
      edge_parent_for_block[pose][ff_edge_end] = edge;
      ff_children[pose][ff_edge_start].push_back(
          std::make_tuple(ff_edge_end, edge));
    }
    for (int edge = 0; edge < max_n_edges_per_ff; ++edge) {
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      if (ff_edge_type == -1) {
        continue;  // break??
      }
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      ff_edge_parent[pose][edge] = edge_parent_for_block[pose][ff_edge_start];
    }
  }
  // deduce root block
  // There is an implicit jump edge from the virtual root of the kinforest to
  // the root of each pose's fold tree. It is okay for multiple edges to come
  // out of the root block and so we talk about the root block and not the root
  // edge.
  std::vector<int> root_block(n_poses, -1);
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int block = 0; block < max_n_blocks; ++block) {
      if (!ff_children[pose][block].empty() && !has_parent[pose][block]) {
        if (root_block[pose] != -1) {
          throw std::runtime_error("Multiple root blocks in fold tree");
        }
        root_block[pose] = block;
        // printf("root_block %d %d\n", pose, block);
      }
    }
  }
  // Now let's perform the depth-first traversals from each pose.
  for (int pose = 0; pose < n_poses; ++pose) {
    int count_dfs_ind = 0;
    std::vector<std::tuple<int, int>> stack;
    for (auto const& child : ff_children[pose][root_block[pose]]) {
      stack.push_back(child);
    }
    while (!stack.empty()) {
      // for (int i = 0; i < stack.size(); ++i) {
      //   printf(
      //       "stack pose %d: iter: %d i: %d %d %d\n",
      //       pose,
      //       count_dfs_ind,
      //       i,
      //       std::get<0>(stack[i]),
      //       std::get<1>(stack[i]));
      // }

      std::tuple<int, int> const child_edge_tuple = stack.back();
      stack.pop_back();
      int const block = std::get<0>(child_edge_tuple);
      int const edge = std::get<1>(child_edge_tuple);
      // printf(
      //     "dfs %d %d %d: e %d (%d %d) w/ %d children\n",
      //     pose,
      //     count_dfs_ind,
      //     block,
      //     edge,
      //     ff_edges_cpu[pose][edge][1],
      //     ff_edges_cpu[pose][edge][2],
      //     ff_children[pose][block].size());
      dfs_order_of_ff_edges[pose][count_dfs_ind] = edge;
      count_dfs_ind += 1;
      for (auto const& child : ff_children[pose][block]) {
        stack.push_back(child);
      }
    }
  }

  // for (int pose = 0; pose < n_poses; ++pose) {
  //   printf("Fold forest children for pose %d\n", pose);
  //   for (int block = 0; block < max_n_blocks; ++block) {
  //     printf("block %d\n", block);
  //     for (auto const& child : ff_children[pose][block]) {
  //       printf("  %d %d\n", std::get<0>(child), std::get<1>(child));
  //     }
  //   }
  // }

  // Step 2:
  // printf("Step 2\n");
  // Step N-10:
  // Write down for each residue the first edge in the fold forest that builds
  // it using the partial order of the fold-forest edges. Note that an edge's
  // start residue is not first built by that edge. In the same traversal, let's
  // also calculate the maximum number of generations of any block type of any
  // edge????? OR let's just assume that every edge has the same number of
  // generations for now and TO DO: write a segmented scan on max() to identify
  // the number of generations for each particular residue that is built by an
  // edge.
  auto first_ff_edge_for_block_cpu_t =
      TPack<Int, 2, Device::CPU>::full({n_poses, max_n_blocks}, -1);
  auto first_ff_edge_for_block_cpu = first_ff_edge_for_block_cpu_t.view;

  auto pose_stack_ff_parent_t =
      TPack<Int, 2, Device::CPU>::full({n_poses, max_n_blocks}, -1);
  auto pose_stack_ff_parent = pose_stack_ff_parent_t.view;

  // auto max_n_gens_for_ff_edge_cpu_t =
  //    TPack<Int, 2, Device::CPU>::zeros({n_poses, max_n_edges_per_ff});
  // auto max_n_gens_for_ff_edge_cpu = max_n_gens_for_ff_edge_cpu_t.view;
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int edge_dfs_ind = 0; edge_dfs_ind < max_n_edges_per_ff;
         ++edge_dfs_ind) {
      int const edge = dfs_order_of_ff_edges[pose][edge_dfs_ind];
      if (edge == -1) {
        break;
      }
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      int const ff_edge_end = ff_edges_cpu[pose][edge][2];
      // printf(
      //     "ff_edge %d %d %d %d %d\n",
      //     pose,
      //     edge,
      //     ff_edge_type,
      //     ff_edge_start,
      //     ff_edge_end);
      // int max_n_gens = 0;
      if (ff_edge_type == 0) {
        int const increment = (ff_edge_start < ff_edge_end) ? 1 : -1;
        // printf("  increment %d\n", increment);
        int const stop = ff_edge_end + increment;
        // printf("  stop %d\n", stop);
        int prev_res = ff_edge_start;
        for (int block = ff_edge_start + increment; block != stop;
             block += increment) {
          // printf("    block %d\n", block);
          first_ff_edge_for_block_cpu[pose][block] = edge;
          pose_stack_ff_parent[pose][block] = prev_res;
          prev_res = block;
          // danger! lives on device -- int const block_type =
          // pose_stack_block_type[pose][block];
        }
      } else if (ff_edge_type == 1) {
        // jump edge! The first block is not built by the jump,
        // but the second block is.
        first_ff_edge_for_block_cpu[pose][ff_edge_end] = edge;
        pose_stack_ff_parent[pose][ff_edge_end] = ff_edge_start;
      }
    }
  }

  // Step 3:
  // printf("Step 3\n");
  // Step N-9:
  // Find the maximum number of generations of any block type of any edge in the
  // fold forest. TEMP!!!
  auto max_n_gens_for_ff_edge_t = TPack<Int, 2, Device::CPU>::full(
      {n_poses, max_n_edges_per_ff}, max_n_gens_per_bt);
  auto max_n_gens_for_ff_edge = max_n_gens_for_ff_edge_t.view;

  // Step 4:
  // printf("Step 4\n");
  // Step N-8:
  // Decompose the fold-forest into paths, minimizing the maximu number of
  // generations. Determine the generational delay of each edge. Then determine
  // the input and output connections for each block. <-- Do  on GPU, entirely
  // parallelizable.
  auto first_child_of_ff_edge_t =
      TPack<Int, 2, Device::CPU>::full({n_poses, max_n_edges_per_ff}, -1);
  auto max_gen_depth_of_ff_edge_t =
      TPack<Int, 2, Device::CPU>::zeros({n_poses, max_n_edges_per_ff});
  auto delay_for_edge_t =
      TPack<Int, 2, Device::CPU>::zeros({n_poses, max_n_edges_per_ff});
  auto first_child_of_ff_edge = first_child_of_ff_edge_t.view;
  auto max_gen_depth_of_ff_edge = max_gen_depth_of_ff_edge_t.view;
  auto delay_for_edge = delay_for_edge_t.view;
  for (int pose = 0; pose < n_poses; ++pose) {
    // traverse edges in reverse order
    for (int edge_in_dfs_ind = n_ff_edges[pose] - 1; edge_in_dfs_ind >= 0;
         edge_in_dfs_ind--) {
      int const edge = dfs_order_of_ff_edges[pose][edge_in_dfs_ind];
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      int const ff_edge_end = ff_edges_cpu[pose][edge][2];
      // printf(
      //     "reverse traversal of ff edge %d %d %d %d\n",
      //     pose,
      //     edge,
      //     ff_edge_start,
      //     ff_edge_end);

      int const ff_edge_max_n_gens = max_n_gens_for_ff_edge[pose][edge];
      int max_child_gen_depth = -1;
      int second_max_child_gen_depth = -1;
      int first_child = -1;
      for (auto const& child : ff_children[pose][ff_edge_end]) {
        int const child_edge = std::get<1>(child);
        int const child_gen_depth = max_gen_depth_of_ff_edge[pose][child_edge];
        // printf(
        //     "Looking at child of res %d: %d %d, max_child_gen_depth %d second
        //     " "max %d\n", ff_edge_end, child_edge, child_gen_depth,
        //     max_child_gen_depth,
        //     second_max_child_gen_depth);
        if (child_gen_depth > max_child_gen_depth) {
          if (max_child_gen_depth != -1) {
            second_max_child_gen_depth = max_child_gen_depth;
          }
          max_child_gen_depth = child_gen_depth;
          first_child = child_edge;
        } else if (child_gen_depth > second_max_child_gen_depth) {
          second_max_child_gen_depth = child_gen_depth;
        }
      }
      first_child_of_ff_edge[pose][edge] = first_child;
      // There are three options for the generational depth of the subtree
      // rooted at this edge, and we take the largest of them:
      // 1. The largest generation depth of any residue built by this edge
      // 2. The largest generation depth of any residue built by the first child
      // of the edge
      // 3. One larger than the largest generation depth of any child besides
      // the first child
      int edge_gen_depth = ff_edge_max_n_gens;
      if (edge_gen_depth < max_child_gen_depth) {
        edge_gen_depth = max_child_gen_depth;
      }
      if (edge_gen_depth < second_max_child_gen_depth + 1) {
        edge_gen_depth = second_max_child_gen_depth + 1;
      }
      // printf(
      //     "max_gen_depth_of_ff_edge %d %d = %d\n", pose, edge,
      //     edge_gen_depth);
      max_gen_depth_of_ff_edge[pose][edge] = edge_gen_depth;
    }

    // for (int i = 0; i < max_n_edges_per_ff; ++i) {
    //   printf(
    //       "first child of %d %d: %d\n",
    //       pose,
    //       i,
    //       first_child_of_ff_edge[pose][i]);
    // }
  }

  // Step 5:
  // printf("Step 5\n");
  // Step N-7:
  // Compute the delay for each edge given the path decomposition of the
  // fold-forest.
  int max_delay = 0;
  for (int pose = 0; pose < n_poses; ++pose) {
    // Now select the first edge to be built from the root block
    // and set the delay for all other edges to 1 and set their
    // parent edge to be the root.
    int max_root_child_gen_depth = -1;
    int max_root_child_edge = -1;
    for (auto const& child : ff_children[pose][root_block[pose]]) {
      int const child_edge = std::get<1>(child);
      int const child_gen_depth = max_gen_depth_of_ff_edge[pose][child_edge];
      if (child_gen_depth > max_root_child_gen_depth) {
        max_root_child_gen_depth = child_gen_depth;
        max_root_child_edge = child_edge;
      }
    }
    delay_for_edge[pose][max_root_child_edge] = 0;
    // We never assigned the first edge to build the root block
    // so let's assign it now. Technically, it's not built by this edge,
    // BUT we need to track the connectivity out of the root somehow, and
    // this will do.

    first_ff_edge_for_block_cpu[pose][root_block[pose]] = max_root_child_edge;
    // printf(
    //     "Root block %d built by edge %d\n",
    //     root_block[pose],
    //     max_root_child_edge);
    for (auto const& child : ff_children[pose][root_block[pose]]) {
      int const child_edge = std::get<1>(child);
      if (child_edge == max_root_child_edge) {
        continue;
      }
      delay_for_edge[pose][child_edge] = 1;
      ff_edge_parent[pose][child_edge] = max_root_child_edge;
      if (max_delay < 1) {
        max_delay = 1;
      }
    }

    for (int edge_in_dfs_ind = 0; edge_in_dfs_ind < n_ff_edges[pose];
         ++edge_in_dfs_ind) {
      int const edge = dfs_order_of_ff_edges[pose][edge_in_dfs_ind];
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      int const ff_edge_end = ff_edges_cpu[pose][edge][2];
      int const first_child = first_child_of_ff_edge[pose][edge];
      int const edge_delay = delay_for_edge[pose][edge];
      for (auto const& child : ff_children[pose][ff_edge_end]) {
        int const child_edge = std::get<1>(child);
        if (child_edge == first_child) {
          delay_for_edge[pose][child_edge] = edge_delay;
        } else {
          delay_for_edge[pose][child_edge] = edge_delay + 1;
          if (max_delay < edge_delay + 1) {
            max_delay = edge_delay + 1;
          }
        }
      }
    }
  }

  // Step 6
  // Step N-6:
  // Construct a topological sort of the fold-forest edges.
  // The sorting is done by edge delay first and then by breadth-
  // first-traversal order of the first edge in each unbroken
  // path of edges and their first descendants, and finally
  // by the order of each edge in the path of edges that builds it
  // E.g. the edge (0,1,2) < (1,0,1) and (0,1,2) < (0,2,0) and
  // (0,2,0) < (1,1,0) and (0, 1, 2) < (0, 1, 3)
  std::vector<std::list<int>> roots_of_subpaths_by_generation(max_delay + 1);
  auto topo_sort_index_for_edge_t =
      TPack<Int, 1, Device::CPU>::full({n_poses * max_n_edges_per_ff}, -1);
  auto topo_sort_index_for_edge = topo_sort_index_for_edge_t.view;
  // Put all the root edges into the roots_of_subpaths_for_generation[0] list
  for (int pose = 0; pose < n_poses; ++pose) {
    // append all the edges coming out of the root block at their given
    // generational delay
    for (auto const& child : ff_children[pose][root_block[pose]]) {
      int const child_edge = std::get<1>(child);
      int const child_gen_delay = delay_for_edge[pose][child_edge];
      roots_of_subpaths_by_generation[child_gen_delay].push_back(
          pose * max_n_edges_per_ff + child_edge);
    }
  }
  // Now let's assign a toplogical sort order to each edge.
  int topo_sort_ind = 0;
  // printf("Max delay: %d\n", max_delay);
  for (int delay = 0; delay < max_delay + 1; ++delay) {
    // printf("Search with Delay = %d\n", delay);
    for (auto const& root_edge : roots_of_subpaths_by_generation[delay]) {
      // printf("Searching path rooted at %d\n", root_edge);
      int const pose = root_edge / max_n_edges_per_ff;

      // // append other children of the root block since they would have been
      // missed. if (delay == 0) {
      //   for (auto const& child_edge_pair :
      //   ff_children[pose][root_block[pose]]) {
      //     int const next_child_edge = std::get<1>(child_edge_pair);
      //     if (next_child_edge != root_edge) {
      //       // Write down this edge as the root of another scan path
      //       // that we will traverse in the next pass
      //       printf("Appending root of subpath %d %d (%d) at delay %d\n",
      //       pose, next_child_edge, pose * max_n_edges_per_ff +
      //       next_child_edge, delay + 1);
      //       roots_of_subpaths_by_generation[delay + 1].push_back(pose *
      //       max_n_edges_per_ff + next_child_edge);
      //     }
      //   }
      // }

      int subpath_root_edge = root_edge % max_n_edges_per_ff;
      while (subpath_root_edge != -1) {
        // Write down the next edge in this path,
        // which we will recusively consider the root of
        // another subpath
        // printf(
        //     "Marking toposort index for edge %d as %d\n",
        //     pose * max_n_edges_per_ff + subpath_root_edge,
        //     topo_sort_ind);
        topo_sort_index_for_edge
            [pose * max_n_edges_per_ff + subpath_root_edge] = topo_sort_ind;
        topo_sort_ind += 1;
        int const first_child = first_child_of_ff_edge[pose][subpath_root_edge];
        // printf("First child %d\n", first_child);
        int const subpath_end_block = ff_edges_cpu[pose][subpath_root_edge][2];
        // printf("Subpath block %d\n", subpath_end_block);
        for (auto const& child_edge_pair :
             ff_children[pose][subpath_end_block]) {
          int const next_child_edge = std::get<1>(child_edge_pair);
          if (next_child_edge != first_child) {
            // Write down this edge as the root of another scan path
            // that we will traverse in the next pass
            // printf(
            //     "Appending root of subpath %d %d (%d) at delay %d\n",
            //     pose,
            //     next_child_edge,
            //     pose * max_n_edges_per_ff + next_child_edge,
            //     delay + 1);
            roots_of_subpaths_by_generation[delay + 1].push_back(
                pose * max_n_edges_per_ff + next_child_edge);
          }
        }
        // Move to the next node in this path
        subpath_root_edge = first_child;
      }

      // int const pose = root_edge / max_n_edges_per_ff;
      // int const edge = root_edge % max_n_edges_per_ff;
      // for (auto const& child :
      // ff_children[pose][ff_edges_cpu[pose][edge][2]]) {
      //   int const child_edge = std::get<1>(child);
      //   int const child_gen_delay = delay_for_edge[pose][child_edge];
      //   roots_of_subpaths_by_generation[delay +
      //   child_gen_delay].push_back(pose * max_n_edges_per_ff + child_edge);
      // }
    }
  }

  return {
      dfs_order_of_ff_edges_t,
      n_ff_edges_t,
      ff_edge_parent_t,
      first_ff_edge_for_block_cpu_t,
      pose_stack_ff_parent_t,
      max_gen_depth_of_ff_edge_t,
      first_child_of_ff_edge_t,
      delay_for_edge_t,
      topo_sort_index_for_edge_t};
}

// P = number of poses
// L = length of the longest pose
// T = number of block types
// A = maximum number of atoms in any block type
// C = maximum number of inter-residue connections in any block type
// E = maximum number of edges in any one FoldTree of the FoldForest
// I = maximum number of input connections in any block type
// O = maximum number of output connections in any block type
// G = maximum number of generations in any block type
// N = maximum number of nodes in any generation in any block type
// S = maximum number of scan path segmentss in any generation in any block type
template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_scans2(
    int64_t const max_n_atoms_per_pose,
    TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
    TView<Int, 2, D> pose_stack_block_type,                 // P x L
    TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
    TView<Int, 3, D>
        ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
    int64_t const max_delay,
    TView<Int, 2, D> delay_for_edge,            // P x E
    TView<Int, 1, D> topo_sort_index_for_edge,  // (P*E)
    TView<Int, 2, D> first_ff_edge_for_block,   // P x L
    TView<Int, 2, D> pose_stack_ff_parent,      // P x L
    // TView<Int, 2, D> pose_stack_ff_conn_to_parent,       // P x L
    TView<Int, 3, D> pose_stack_block_in_and_first_out,  // P x L x 2
    TView<Int, 3, D> block_type_parents,                 // T x O x A
    TView<Int, 2, D> kfo_2_orig_mapping,                 // K x 3
    TView<Int, 3, D> atom_kfo_index,                     // P x L x A
    TView<Int, 1, D> block_type_jump_atom,               // T
    TView<Int, 1, D> block_type_n_conn,                  // T
    TView<Int, 2, D>
        block_type_polymeric_conn_index,  // T x 2 - 2 is for "down" and "up"
                                          // connections.
    TView<Int, 3, D> block_type_n_gens,   // T x I x O
    TView<Int, 5, D> block_type_kts_conn_info,  // T x I x O x C x 2 - 2 is for
                                                // gen (0) and scan-path-seg (1)
    TView<Int, 5, D> block_type_nodes_for_gens,          // T x I x O x G x N
    TView<Int, 4, D> block_type_n_scan_path_segs,        // T x I x O x G
    TView<Int, 5, D> block_type_scan_path_seg_starts,    // T x I x O x G x S
    TView<bool, 5, D> block_type_scan_path_seg_is_real,  // T x I x O x G x S
    TView<bool, 5, D>
        block_type_scan_path_seg_is_inter_block,      // T x I x O x G x S
    TView<Int, 5, D> block_type_scan_path_seg_length  // T x I x O x G x S
    )
    -> std::tuple<
        TPack<Int, 1, D>,
        TPack<Int, 1, D>,
        TPack<Int, 2, D>,
        TPack<Int, 1, D>,
        TPack<Int, 1, D>,
        TPack<Int, 2, D>> {
  // The final step is to construct the nodes, scans, and gens tensors
  // from the per-block-type stencils.
  //

  // For each block, we need to know which FoldForest edge builds it.
  // For each FF edge, we need to know its generational delay.
  // With that, we can calculate the generational delay for each block.
  // For each block-scan-path, we need to know its offset into the nodes
  // tensor. For each block-scan path, we need to know its offset into the
  // block-scans list. Then we can ask each block-scan path how many nodes it
  // has, and generate the offset using scan. We need to know how many
  // block scan paths there are. We need to map block-scan path index
  // to block, generation, and scan-within-the-generation.

  // In order to know the block-scan-path index for any block-scan path, we
  // have to
  // count the number of block-scan paths that come before it. This can be
  // tricky
  // because some block-scan paths continue into other blocks, and we do
  // not know
  // a priori how many block-scan paths there are downstream of such a
  // block-scan path.
  // For each (inter-block) scan path, we have to calculate how many
  // block-scan paths
  // comprise it. Each scan path can be readily identified from the fold
  // forest.
  // Each block type should identify which scan paths are inter-block so
  // it's easy to
  // figure out for each block-scan path extend into other blocks: not all
  // do.

  // Step N-5:

  // Step N-4: count the number of blocks that build each
  // (perhaps-multi-res) scan path.

  // Step N-3: perform a segmented scan on the number of blocks that build
  // each
  // (perhaps-multi-res) scan path.

  // Step N-2: write the number of atoms in each scan path to the
  // appropriate place
  // in the n_atoms_for_scan_path_for_gen tensor.

  // Step N-1: perform a scan on the number of atoms in each scan path to
  // get the
  // nodes tensor offset.

  // Step N: copy the scan path stencils into the nodes tensor, adding the
  // pose-stack- and block- offsets to the atom indices. Note that the
  // upstream
  // jump atom must be added for jump edges that are the roots of paths.

  // Note that gens_fw and gen_bw will both be on the device and must be
  // moved to the CPU.
  using namespace score::common;
  LAUNCH_BOX_32;

  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_edges_per_ff = ff_edges.size(1);
  int const max_n_input_conn = block_type_kts_conn_info.size(1);
  int const max_n_output_conn = block_type_kts_conn_info.size(1);
  int const max_n_gens_per_bt = block_type_nodes_for_gens.size(3);
  // How many generations of segmented scan we will actually be performing
  // It represents the multiple generations that any one block type requires
  // as well as the generation delay that edges in the FoldForest can have.
  int const n_gens_total = max_n_gens_per_bt + max_delay + 1;
  int const max_n_nodes_per_gen = block_type_nodes_for_gens.size(4);
  int const max_n_scan_path_segs_per_gen =
      block_type_scan_path_seg_starts.size(4);
  // printf("n_poses %d\n", n_poses);
  // printf("max_n_blocks %d\n", max_n_blocks);
  // printf("max_n_edges_per_ff %d\n", max_n_edges_per_ff);
  // printf("max_n_input_conn %d\n", max_n_input_conn);
  // printf("max_n_output_conn %d\n", max_n_output_conn);
  // printf("max_n_gens_per_bt %d\n", max_n_gens_per_bt);
  // printf("max_n_nodes_per_gen %d\n", max_n_nodes_per_gen);
  // printf("max_n_scan_path_segs_per_gen %d\n", max_n_scan_path_segs_per_gen);

  auto n_kin_atoms_offset_for_block_t =
      TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto n_sps_for_ffedge_for_gen_by_topo_sort_t =
      TPack<Int, 2, D>::zeros({n_gens_total, n_poses * max_n_edges_per_ff});
  auto n_sps_for_ffedge_for_gen_segment_starts_t =
      TPack<Int, 1, D>::zeros({n_gens_total});
  // auto sp_offset_for_ffedge_for_gen_by_topo_sort_t =
  //     TPack<Int, 2, D>::zeros({max_n_gens, n_poses * max_n_edges_per_ff});
  auto n_kin_atoms_offset_for_block = n_kin_atoms_offset_for_block_t.view;
  auto n_sps_for_ffedge_for_gen_by_topo_sort =
      n_sps_for_ffedge_for_gen_by_topo_sort_t.view;
  auto n_sps_for_ffedge_for_gen_segment_starts =
      n_sps_for_ffedge_for_gen_segment_starts_t.view;

  // Step 6:
  // Determine the roots of the forward and backwards scan paths.
  // For each edge, we will determine whether it's the root of a
  // forward scan path by looking at its delay and the delay for the
  // edge that builds the start block and if they are the same delay
  // then the edge is built as a continuation of the path that goes
  // through its parent. That means this edge is not the root of a
  // scan path and it also means that the parent is not the root of
  // a backwards scan path. If the delays are different, then the
  // edge is the root of a scan path and, while seeing that the delays
  // are different means that this edge must have an "older sibling"
  // that is the continuation of the path of the parent edge, we
  // will not mark the parent as being not a backwards-scan-path
  // root but instead leave that marking to the iteration when we
  // examine the older sibling. We start by marking no edges as
  // roots of forward scan paths and proceed to mark them as we
  // iterate; we start by marking all edges as roots of backwards-
  // scan paths and proceed to eliminate them as we iterate.
  //
  // Along the way, we will encounter exactly one edge for each
  // tree in this forest that is labeled as having itself as its
  // parent (or, rather, an edge that is the first edge to build
  // the start block) and this edge is the root of the fold tree.
  // Note the terminology difference: "scan path" vs "scan path
  // segment".

  // printf("Step 6a\n");
  auto is_edge_end_block_scan_path_seg_root_of_bw_scan_path_t =
      TPack<Int, 4, D>::zeros(
          {n_poses,
           max_n_blocks,
           max_n_gens_per_bt,
           max_n_scan_path_segs_per_gen});
  auto is_edge_end_block_scan_path_seg_root_of_bw_scan_path =
      is_edge_end_block_scan_path_seg_root_of_bw_scan_path_t.view;
  auto mark_ff_edge_end_block_output_conns_as_potential_bw_sp_roots =
      ([=] TMOL_DEVICE_FUNC(int i) {
        int const pose = i / max_n_edges_per_ff;
        int const edge = i % max_n_edges_per_ff;
        int const ff_edge_type = ff_edges[pose][edge][0];
        if (ff_edge_type == -1) {
          // Sentinel value: this is not a real edge
          return;
        }
        int const ff_edge_start = ff_edges[pose][edge][1];
        int const first_edge_for_start = first_ff_edge_for_block
            [pose][ff_edge_start];  // what edge first builds the start residue?
        int const ff_edge_end = ff_edges[pose][edge][2];
        int const end_bt = pose_stack_block_type[pose][ff_edge_end];
        // printf("pose %d edge %d end_bt %d\n", pose, edge, end_bt);
        int const end_bt_n_conn = block_type_n_conn[end_bt];
        // printf("n_conn: %d\n", end_bt_n_conn);
        int const end_in_conn =
            pose_stack_block_in_and_first_out[pose][ff_edge_end][0];
        int const end_out_conn =
            pose_stack_block_in_and_first_out[pose][ff_edge_end][1];
        for (int j = 0; j < end_bt_n_conn; ++j) {
          if (j == end_in_conn || j == end_out_conn) {
            continue;
          }
          int const j_gen =
              block_type_kts_conn_info[end_bt][end_in_conn][end_out_conn][j][0];
          int const j_sps =
              block_type_kts_conn_info[end_bt][end_in_conn][end_out_conn][j][1];
          if (j_gen == -1) {
            // If we have a leaf of the fold forest, then all scan path segments
            // will be roots of backwards scan paths.
            continue;
          }
          // printf(
          //     "Possible root of bw scan path: pose %d block %d j %d j_gen %d
          //     " "j_sps %d\n", pose, ff_edge_end, j, j_gen, j_sps);
          is_edge_end_block_scan_path_seg_root_of_bw_scan_path[pose]
                                                              [ff_edge_end]
                                                              [j_gen][j_sps] =
                                                                  true;
        }
        if (first_edge_for_start == edge) {
          // this is the root of the fold tree
          int start_bt = pose_stack_block_type[pose][ff_edge_start];
          int const start_bt_n_conn = block_type_n_conn[start_bt];
          int const start_in_conn =
              pose_stack_block_in_and_first_out[pose][ff_edge_start]
                                               [0];  // should be "root"
                                                     // connection; n_conn + 1
          int const start_out_conn =
              pose_stack_block_in_and_first_out[pose][ff_edge_start][1];
          // printf(
          //     "Step 6a: Root edge. start_bt %d start_in_conn %d
          //     start_out_conn "
          //     "%d\n",
          //     start_bt,
          //     start_in_conn,
          //     start_out_conn);
          for (int j = 0; j < start_bt_n_conn; ++j) {
            if (j == start_in_conn || j == start_out_conn) {
              continue;
            }
            int const j_gen = block_type_kts_conn_info[start_bt][start_in_conn]
                                                      [start_out_conn][j][0];
            int const j_sps = block_type_kts_conn_info[start_bt][start_in_conn]
                                                      [start_out_conn][j][1];
            if (j_gen == -1) {
              // If we have a leaf of the fold forest, then all scan path
              // segments will be roots of backwards scan paths.
              continue;
            }
            // printf(
            //     "Tree root: Possible root of bw scan path: pose %d block %d j
            //     "
            //     "%d j_gen %d "
            //     "j_sps %d\n",
            //     pose,
            //     ff_edge_start,
            //     j,
            //     j_gen,
            //     j_sps);
            is_edge_end_block_scan_path_seg_root_of_bw_scan_path[pose]
                                                                [ff_edge_start]
                                                                [j_gen][j_sps] =
                                                                    true;
          }
        }
      });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_edges_per_ff,
      mark_ff_edge_end_block_output_conns_as_potential_bw_sp_roots);
  gpuErrPeek;
  gpuErrSync;

  // printf("Step 6\n");
  auto is_ff_edge_root_of_scan_path_t =
      TPack<bool, 2, D>::zeros({n_poses, max_n_edges_per_ff});
  auto is_ff_edge_root_of_fold_tree_t =
      TPack<bool, 2, D>::zeros({n_poses, max_n_edges_per_ff});
  auto is_ff_edge_root_of_scan_path_bw_t =
      TPack<bool, 2, D>::ones({n_poses, max_n_edges_per_ff});
  // auto is_ff_edge_root_of_fold_tree_bw_t =
  //     TPack<bool, 2, D>::zeros({n_poses, max_n_edges_per_ff}); // ??

  auto is_ff_edge_root_of_scan_path = is_ff_edge_root_of_scan_path_t.view;
  auto is_ff_edge_root_of_fold_tree = is_ff_edge_root_of_fold_tree_t.view;
  auto is_ff_edge_root_of_scan_path_bw = is_ff_edge_root_of_scan_path_bw_t.view;
  auto mark_ff_edge_as_root_of_scan_path = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / max_n_edges_per_ff;
    int const edge = i % max_n_edges_per_ff;
    int const ff_edge_type = ff_edges[pose][edge][0];
    if (ff_edge_type == -1) {
      // Sentinel value: this is not a real edge
      return;
    }
    int const ff_edge_start = ff_edges[pose][edge][1];
    int const ff_edge_end = ff_edges[pose][edge][2];
    int const first_edge_for_start =
        first_ff_edge_for_block[pose][ff_edge_start];
    // printf(
    //     "edge %d's edge start %d has first edge for start %d\n",
    //     edge,
    //     ff_edge_start,
    //     first_edge_for_start);
    if (edge == first_edge_for_start) {
      // we are looking at the root of the fold tree
      is_ff_edge_root_of_fold_tree[pose][edge] = true;
      is_ff_edge_root_of_scan_path[pose][edge] = true;
    } else {
      int const ff_edge_delay = delay_for_edge[pose][edge];
      int const first_edge_delay = delay_for_edge[pose][first_edge_for_start];
      if (ff_edge_delay != first_edge_delay) {
        // this edge is not the first child of the parent edge
        // which means it must root its own scan path
        // printf(
        //     "edge %d delay %d vs first-edge-for-start %d first edge delay
        //     %d\n", edge, ff_edge_delay, first_edge_for_start,
        //     first_edge_delay);
        // printf("edge %d on pose %d is root of scan path\n", edge, pose);
        is_ff_edge_root_of_scan_path[pose][edge] = true;

        // Find the SPS on the end block of first_edge_for_start / start
        // block of "edge" that connects it to the next residue on the edge
        // and mark it as NOT being a root of a backwards scan path.
        int const start_bt = pose_stack_block_type[pose][ff_edge_start];
        if (ff_edge_type == 1) {
          // jump edge: noop
        } else {
          // bond edge: are we going from N->C or C->N?
          int const conn_ind = (ff_edge_start < ff_edge_end) ? 1 : 0;
          int const in_conn =
              pose_stack_block_in_and_first_out[pose][ff_edge_start][0];
          int const out_conn =
              pose_stack_block_in_and_first_out[pose][ff_edge_start][1];
          int const gen = block_type_kts_conn_info[start_bt][in_conn][out_conn]
                                                  [conn_ind][0];
          int const sps = block_type_kts_conn_info[start_bt][in_conn][out_conn]
                                                  [conn_ind][1];
          if (gen != -1) {
            // printf(
            //     "Eliminating sps %d %d %d %d as root of bw scan path\n",
            //     pose,
            //     ff_edge_start,
            //     gen,
            //     sps);
            is_edge_end_block_scan_path_seg_root_of_bw_scan_path[pose]
                                                                [ff_edge_start]
                                                                [gen][sps] =
                                                                    false;
          }
        }

      } else {
        // the parent edge continues on into this edge
        // so mark "first_edge_for_start" as not a root of a backwards
        // scan path; "edge" may still be a root, we don't know!
        // printf(
        //     "edge %d on pose %d is not root of bw scan path\n",
        //     first_edge_for_start,
        //     pose);
        is_ff_edge_root_of_scan_path_bw[pose][first_edge_for_start] = false;
        // Find the SPS on the end block of first_edge_for_start / start
        // block of "edge" that connects it to the next residue on the edge
        // and mark it as NOT being a root of a backwards scan path.
        int const start_bt = pose_stack_block_type[pose][ff_edge_start];
        if (ff_edge_type == 1) {
          // jump edge: noop
        } else {
          // bond edge: are we going from N->C or C->N?
          int const conn_ind = (ff_edge_start < ff_edge_end) ? 1 : 0;
          int const in_conn =
              pose_stack_block_in_and_first_out[pose][ff_edge_start][0];
          int const out_conn =
              pose_stack_block_in_and_first_out[pose][ff_edge_start][1];
          int const gen = block_type_kts_conn_info[start_bt][in_conn][out_conn]
                                                  [conn_ind][0];
          int const sps = block_type_kts_conn_info[start_bt][in_conn][out_conn]
                                                  [conn_ind][1];
          if (gen != -1) {
            // printf(
            //     "Eliminating sps %d %d %d %d as root of bw scan path\n",
            //     pose,
            //     ff_edge_start,
            //     gen,
            //     sps);
            is_edge_end_block_scan_path_seg_root_of_bw_scan_path[pose]
                                                                [ff_edge_start]
                                                                [gen][sps] =
                                                                    false;
          }
        }
      }
    }

    // printf(
    //     "is_ff_edge_root_of_scan_path[%d][%d] = %d\n",
    //     pose,
    //     edge,
    //     is_ff_edge_root_of_scan_path[pose][edge]);
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_edges_per_ff, mark_ff_edge_as_root_of_scan_path);
  gpuErrPeek;
  gpuErrSync;

  // Step 7
  // Step N-5:
  // Mark the scan-path segments that root each (jump & non-jump) fold-forest
  // edge. This will store the per-pose indexing of the fold-forest edge rather
  // than the global indexing, but they can be interconverted easily:
  // pose_ff_edge_index = global_edge_index % max_n_edges_per_ff
  // global_edge_index = pose * max_n_edges_per_ff + pose_ff_edge_index
  // printf("Step 7\n");

  auto non_jump_ff_edge_rooted_at_scan_path_seg_t = TPack<Int, 4, D>::full(
      {n_poses, max_n_blocks, max_n_gens_per_bt, max_n_scan_path_segs_per_gen},
      -1);
  auto non_jump_ff_edge_rooted_at_scan_path_seg =
      non_jump_ff_edge_rooted_at_scan_path_seg_t.view;
  auto non_jump_ff_edge_rooted_at_scan_path_seg_bw_t = TPack<Int, 4, D>::full(
      {n_poses, max_n_blocks, max_n_gens_per_bt, max_n_scan_path_segs_per_gen},
      -1);
  auto non_jump_ff_edge_rooted_at_scan_path_seg_bw =
      non_jump_ff_edge_rooted_at_scan_path_seg_bw_t.view;
  auto jump_ff_edge_rooted_at_scan_path_seg_t = TPack<Int, 4, D>::full(
      {n_poses, max_n_blocks, max_n_gens_per_bt, max_n_scan_path_segs_per_gen},
      -1);
  auto jump_ff_edge_rooted_at_scan_path_seg =
      jump_ff_edge_rooted_at_scan_path_seg_t.view;
  auto jump_ff_edge_rooted_at_scan_path_seg_bw_t = TPack<Int, 4, D>::full(
      {n_poses, max_n_blocks, max_n_gens_per_bt, max_n_scan_path_segs_per_gen},
      -1);
  auto jump_ff_edge_rooted_at_scan_path_seg_bw =
      jump_ff_edge_rooted_at_scan_path_seg_bw_t.view;

  // Unclear if this data is necessary: it seems to mirror exactly the
  // data in jump_ff_edge_rooted_At_scan_path_seg.
  // auto jump_ff_edge_rooted_at_scan_path_seg_bw_t = TPack<Int, 4, D>::full(
  //     {n_poses, max_n_blocks, max_n_gens_per_bt,
  //     max_n_scan_path_segs_per_gen}, -1);
  // auto jump_ff_edge_rooted_at_scan_path_seg_bw =
  //     jump_ff_edge_rooted_at_scan_path_seg_bw_t.view;

  auto mark_scan_path_segs_that_root_fold_forest_edges = ([=] TMOL_DEVICE_FUNC(
                                                              int i) {
    int const pose = i / max_n_edges_per_ff;
    int const edge = i % max_n_edges_per_ff;
    int const ff_edge_type = ff_edges[pose][edge][0];
    if (ff_edge_type == -1) {
      // Not an actual edge of the fold tree
      return;
    }
    bool const is_root = is_ff_edge_root_of_fold_tree[pose][edge];
    int const ff_edge_start = ff_edges[pose][edge][1];
    int const ff_edge_end = ff_edges[pose][edge][2];
    if (ff_edge_type == 1) {
      // Jump edge
      // A jump edge uses only one atom of the start block
      // and we will append that atom to the nodes list for
      // the first scan path of the end block. We need not
      // look up the scan path on the end block that builds
      // this edge because it will always be the first, but
      // we do need to know whether we are looking at the root
      // of the fold tree.
      int const start_block_first_edge =
          first_ff_edge_for_block[pose][ff_edge_start];
      if (edge == start_block_first_edge) {
        // we are looking at the root of the fold tree
        // printf(
        //     "Jump edge %d on pose %d is root of fold tree; ff_edge_start %d "
        //     "ff_edge_end %d; is root %d\n",
        //     edge,
        //     pose,
        //     ff_edge_start,
        //     ff_edge_end,
        //     is_root);
        jump_ff_edge_rooted_at_scan_path_seg[pose][ff_edge_start][0][0] = edge;
        jump_ff_edge_rooted_at_scan_path_seg_bw[pose][ff_edge_end][0][0] = edge;
      } else {
        // printf(
        //     "Jump edge %d on pose %d is not root of fold tree; ff_edge_start
        //     "
        //     "%d ff_edge_end %d; is root %d\n",
        //     edge,
        //     pose,
        //     ff_edge_start,
        //     ff_edge_end,
        //     is_root);
        jump_ff_edge_rooted_at_scan_path_seg[pose][ff_edge_end][0][0] = edge;
        jump_ff_edge_rooted_at_scan_path_seg_bw[pose][ff_edge_end][0][0] = edge;
      }

    } else {
      int const start_block_type = pose_stack_block_type[pose][ff_edge_start];
      int const start_block_in =
          pose_stack_block_in_and_first_out[pose][ff_edge_start][0];
      int const start_block_out =
          pose_stack_block_in_and_first_out[pose][ff_edge_start][1];
      int const start_block_type_out_conn_ind =
          block_type_polymeric_conn_index[start_block_type]
                                         [(ff_edge_start < ff_edge_end) ? 1
                                                                        : 0];

      int const exiting_scan_path_seg_gen =
          block_type_kts_conn_info[start_block_type][start_block_in]
                                  [start_block_out]
                                  [start_block_type_out_conn_ind][0];
      int const exiting_scan_path_seg =
          block_type_kts_conn_info[start_block_type][start_block_in]
                                  [start_block_out]
                                  [start_block_type_out_conn_ind][1];
      // printf(
      //     "for edge (%d, %d - %d), start_block_in %d start_block_out %d, "
      //     "conn_ind %d\n",
      //     ff_edge_start,
      //     ff_edge_end,
      //     ff_edge_type,
      //     start_block_in,
      //     start_block_out,
      //     start_block_type_out_conn_ind);
      // printf(
      //     "non_jump_ff_edge_rooted_at_scan_path[%d][%d][%d][%d] = %d\n",
      //     pose,
      //     ff_edge_start,
      //     exitting_scan_path_seg_gen,
      //     exitting_scan_path_seg,
      //     (pose * max_n_edges_per_ff + edge));
      if (is_root || exiting_scan_path_seg_gen != 0) {
        non_jump_ff_edge_rooted_at_scan_path_seg[pose][ff_edge_start]
                                                [exiting_scan_path_seg_gen]
                                                [exiting_scan_path_seg] = edge;
        non_jump_ff_edge_rooted_at_scan_path_seg_bw[pose][ff_edge_end][0][0] =
            edge;
      } else {
        // This edge exits the ff_edge_start residue through the primary
        // exit path. Therefore, when we are later looking to identify the
        // SPS that roots this edge so that we can make sure that we treat
        // it as the root of a scan path, we do not have to: the edge that
        // builds this residue will either be the root of a scan path, or
        // it will not, but in any event, this edge will not.
        // NO.
        // Okay, so, we have the edge that builds the start residue already;
        // and the SPS for this residue already brings us to the connection
        // atom for the next residue, so let's just say that the edge is
        // "rooted" at the start + increment residue, with increment being
        // 1 for N->C and -1 for C->N.
        // We know that it's the generation 0 SPS for the next residue,
        // so we don't have to look that up.
        int const increment = (ff_edge_start < ff_edge_end) ? 1 : -1;
        int const next_residue = ff_edge_start + increment;
        non_jump_ff_edge_rooted_at_scan_path_seg[pose][next_residue][0][0] =
            edge;
        non_jump_ff_edge_rooted_at_scan_path_seg_bw[pose][ff_edge_end][0][0] =
            edge;
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_edges_per_ff,
      mark_scan_path_segs_that_root_fold_forest_edges);
  gpuErrPeek;
  gpuErrSync;

  // Step 8
  // Step N-4:
  // Count the number of scan-path segs that build each ff-edge for
  // each generation with edges ordered by their topological-sort index
  // printf("Step 8\n");
  auto n_blocks_that_build_tsedge_for_gen_tp =
      TPack<Int, 1, D>::zeros({n_poses * max_n_edges_per_ff * n_gens_total});
  auto n_blocks_that_build_tsedge_for_gen =
      n_blocks_that_build_tsedge_for_gen_tp.view;
  auto n_blocks_that_build_tsedge_for_gen_bw_tp =
      TPack<Int, 1, D>::zeros({n_poses * max_n_edges_per_ff * n_gens_total});
  auto n_blocks_that_build_tsedge_for_gen_bw =
      n_blocks_that_build_tsedge_for_gen_bw_tp.view;
  auto count_n_blocks_for_ffedge_for_gen_by_topo_sort =
      ([=] TMOL_DEVICE_FUNC(int ind) {
        int i = ind;
        int const pose = i / (max_n_gens_per_bt * max_n_edges_per_ff);
        i = i - pose * (max_n_gens_per_bt * max_n_edges_per_ff);
        int const edge = i / max_n_gens_per_bt;
        int const gen = i % max_n_gens_per_bt;

        int const edge_type = ff_edges[pose][edge][0];
        if (edge_type == -1) {
          return;
        }
        // Look, we can be extra generous and allocate space
        // for a block that is not truly built by this edge,
        // if, e.g., the edge is a jump and the block would have
        // already been built by another edge.
        int const ff_edge_start = ff_edges[pose][edge][1];
        int const ff_edge_end = ff_edges[pose][edge][2];
        int const n_blocks =
            (edge_type == 0 ? (
                 ff_edge_end > ff_edge_start ? ff_edge_end - ff_edge_start + 1
                                             : ff_edge_start - ff_edge_end + 1)
                            : 2);
        int const edge_delay = delay_for_edge[pose][edge];
        int const ff_edge_gen = gen + edge_delay;
        int const ff_edge_gen_bw = (n_gens_total - 1) - ff_edge_gen;
        int const edge_toposort_index =
            topo_sort_index_for_edge[pose * max_n_edges_per_ff + edge];
        int const edge_toposort_index_bw =
            n_poses * max_n_edges_per_ff - 1 - edge_toposort_index;
        // printf(
        //     "edge_toposort_index %d edge_toposort_index_bw %d\n",
        //     edge_toposort_index,
        //     edge_toposort_index_bw);

        n_blocks_that_build_tsedge_for_gen
            [ff_edge_gen * n_poses * max_n_edges_per_ff + edge_toposort_index] =
                n_blocks;
        n_blocks_that_build_tsedge_for_gen_bw
            [ff_edge_gen_bw * n_poses * max_n_edges_per_ff
             + edge_toposort_index_bw] = n_blocks;
      });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_edges_per_ff * max_n_gens_per_bt,
      count_n_blocks_for_ffedge_for_gen_by_topo_sort);

  // auto count_n_segs_for_ffedge_for_gen_by_topo_sort =
  //     ([=] TMOL_DEVICE_FUNC(int ind) {
  //       int i = ind;
  //       int const pose =
  //           i / (max_n_blocks * max_n_gens_per_bt *
  //           max_n_scan_paths_per_gen);
  //       i = i - pose * max_n_blocks * max_n_gens_per_bt *
  //       max_n_scan_paths_per_gen; int const block = i / (max_n_gens_per_bt *
  //       max_n_scan_paths_per_gen); i = i - block * max_n_gens_per_bt *
  //       max_n_scan_paths_per_gen; int const gen = i /
  //       max_n_scan_paths_per_gen; int const scan_path = i %
  //       max_n_scan_paths_per_gen;
  //       // printf("count_n_segs_for_ffedge_for_gen_by_topo_sort %d %d %d %d
  //       %d\n",
  //       //       ind,
  //       //       pose,
  //       //       block,
  //       //       gen,
  //       //       scan_path
  //       // );
  //       // if (i < n_gens_total) {
  //       //   // Need indices of the start of each segment for each gen for
  //       //   // seg-scan.
  //       //   n_sps_for_ffedge_for_gen_segment_starts[i] =
  //       //       i * n_poses * max_n_edges_per_ff;
  //       // }

  //       int const block_type = pose_stack_block_type[pose][block];
  //       if (block_type == -1) {
  //         return;
  //       }
  //       int const block_type_in =
  //       pose_stack_block_in_and_first_out[pose][block][0]; int const
  //       block_type_out = pose_stack_block_in_and_first_out[pose][block][1];
  //       if (scan_path >=
  //       block_type_n_scan_paths[block_type][block_type_in][block_type_out][gen])
  //       {
  //         // printf("count_n_segs_for_ffedge_for_gen_by_topo_sort early exit
  //         %d vs %d \n", scan_path,
  //         block_type_n_scan_paths[block_type][block_type_in][block_type_out][gen]);
  //         return;
  //       }
  //       int ff_edge = first_ff_edge_for_block[pose][block];
  //       int const ff_edge_rooted_at_scan_path =
  //           non_jump_ff_edge_rooted_at_scan_path[pose][block][gen][scan_path];
  //       if (ff_edge_rooted_at_scan_path != -1) {
  //         // printf("ff_edge_rooted_at_scan_path: %d\n",
  //         ff_edge_rooted_at_scan_path); ff_edge =
  //         ff_edge_rooted_at_scan_path;
  //       }
  //       int const global_ff_edge_index = pose * max_n_edges_per_ff + ff_edge;
  //       // printf("ffedge %d\n", ff_edge);
  //       int const ff_edge_delay = delay_for_edge[pose][ff_edge];
  //       // printf("ffedge delay %d\n", ff_edge_delay);
  //       int const ff_edge_topo_sort_index =
  //           topo_sort_index_for_edge[global_ff_edge_index];
  //       // printf("ffedge topo sort index %d\n", ff_edge_topo_sort_index);
  //       // now we can increment the number of scan paths that build this edge
  //       printf("block %d %d, scan path %d, incrementing n sps for ffedge %d
  //       (%d %d) ff_edge_topo_sort_index %d\n", pose, block, scan_path,
  //       ff_edge, gen, ff_edge_delay, ff_edge_topo_sort_index); accumulate<D,
  //       Int>::add(
  //           n_blocks_that_build_edge_for_gen[(gen + ff_edge_delay) *
  //           max_n_edges_per_ff * n_poses + ff_edge_topo_sort_index], 1);
  //     });
  // DeviceDispatch<D>::template forall<launch_t>(
  //     n_poses * max_n_blocks * max_n_gens_per_bt,
  //     count_n_segs_for_ffedge_for_gen_by_topo_sort);

  // for (int gen = 0; gen < n_gens_total; ++gen) {
  //   for (int edge = 0; edge < max_n_edges_per_ff * n_poses; ++edge) {
  //     printf(
  //         "n_blocks_that_build_tsedge_for_gen[%d][%d] = %d\n",
  //         gen,
  //         edge,
  //         n_blocks_that_build_tsedge_for_gen
  //             [gen * max_n_edges_per_ff * n_poses + edge]);
  //   }
  // }

  // Step 10
  // Step N-3:
  // Now, run scan on n_blocks_that_build_edge_for_gen to get
  // block_offset_for_tsedge_for_gen
  // printf("Step 10\n");
  int const n_gens_x_n_edges = n_gens_total * n_poses * max_n_edges_per_ff;
  auto block_offset_for_tsedge_for_gen_tp =
      TPack<Int, 1, D>::zeros({n_gens_x_n_edges});
  auto block_offset_for_tsedge_for_gen =
      block_offset_for_tsedge_for_gen_tp.view;
  auto block_offset_for_tsedge_for_gen_bw_tp =
      TPack<Int, 1, D>::zeros({n_gens_x_n_edges});
  auto block_offset_for_tsedge_for_gen_bw =
      block_offset_for_tsedge_for_gen_bw_tp.view;

  // SCAN!
  int n_blocks_building_edges_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_blocks_that_build_tsedge_for_gen.data(),
          block_offset_for_tsedge_for_gen.data(),
          n_gens_total * n_poses * max_n_edges_per_ff,
          mgpu::plus_t<Int>());
  // second scan for backward pass
  int n_blocks_building_edges_total2 =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_blocks_that_build_tsedge_for_gen_bw.data(),
          block_offset_for_tsedge_for_gen_bw.data(),
          n_gens_total * n_poses * max_n_edges_per_ff,
          mgpu::plus_t<Int>());

  // printf("n_blocks_building_edges_total %d\n",
  // n_blocks_building_edges_total);
  auto is_scan_path_seg_root_of_scan_path_t = TPack<Int, 1, D>::full(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen}, 0);
  auto is_scan_path_seg_root_of_scan_path_bw_t = TPack<Int, 1, D>::full(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen}, 0);
  auto is_scan_path_seg_root_of_scan_path =
      is_scan_path_seg_root_of_scan_path_t.view;
  auto is_scan_path_seg_root_of_scan_path_bw =
      is_scan_path_seg_root_of_scan_path_bw_t.view;

  // for (int ind = 0; ind < n_gens_total * n_poses * max_n_edges_per_ff; ++ind)
  // {
  //   int i = ind;
  //   int const pose = i / (n_gens_total * max_n_edges_per_ff);
  //   i = i - pose * n_gens_total * max_n_edges_per_ff;
  //   int const edge = i / (n_gens_total);
  //   i = i - edge * n_gens_total;
  //   int const gen = i % n_gens_total;

  //   printf(
  //       "block_offset_for_tsedge_for_gen[(%d, %d, %d) = %d] = %d\n",
  //       pose,
  //       edge,
  //       gen,
  //       ind,
  //       block_offset_for_tsedge_for_gen[ind]);
  // }
  // auto sp_offset_for_ff_edge_for_gen_by_topo_sort_tp =
  //     DeviceDispatch<D>::template segmented_scan<mgpu::scan_type_exc>(
  //         n_sps_for_ffedge_for_gen_by_topo_sort.data(),
  //         n_sps_for_ffedge_for_gen_segment_starts.data(),
  //         n_poses * max_n_edges_per_ff * (max_n_gens + max_delay + 1),
  //         (max_n_gens + max_delay + 1),
  //         mgpu::plus_t<Int>(),
  //         Int(0));
  // auto sp_offset_for_ff_edge_for_gen_by_topo_sort =
  //     sp_offset_for_ff_edge_for_gen_by_topo_sort_tp.view;
  // for (int ind = 0; ind < n_poses * max_n_edges_per_ff * (max_n_gens +
  // max_delay + 1); ++ind) {
  //   printf("sp_offset_for_ff_edge_for_gen_by_topo_sort[%d] = %d\n",
  //          ind,
  //          sp_offset_for_ff_edge_for_gen_by_topo_sort[ind]);
  // }

  // convenience function for determining the rank of a block within the
  // fold-forest edge that builds it.
  auto polymer_edge_index_for_block =
      ([=] TMOL_DEVICE_FUNC(
           TView<Int, 3, D> const& ff_edges,
           int pose,
           int edge_on_pose,
           int block) -> std::tuple<int, int> {
        // For a polymer edge (peptide edge), return the index of a particular
        // block on that edge; e.g., for the edge 10->25, block 15 is at index
        // 5,        and for the edge 25->10, block 24 is at index 1.
        int const ff_start_block = ff_edges[pose][edge_on_pose][1];
        int const ff_end_block = ff_edges[pose][edge_on_pose][2];
        if (ff_start_block < ff_end_block) {
          return {block - ff_start_block, ff_end_block - block};
        } else {
          return {ff_start_block - block, block - ff_end_block};
        }
      });

  // Step 11
  // Step N-2:
  // Alright, now let's write down the number of atoms for each scan path seg
  // for each generation. UNSURE IF NEXT STEP NEEDED: While we're at it, record
  // the number of atoms for each real block so we can calculate the kin-atom
  // offset. Block (0,0) will say it holds natoms(0,0) + 1 to account for the
  // root of the kinforest, node "0."
  // printf("Step 11\n");
  auto n_atoms_for_scan_path_seg_for_gen_t = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto n_atoms_for_scan_path_seg_for_gen_bw_t = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto n_scan_paths_for_gen_t = TPack<Int, 1, D>::zeros({n_gens_total + 1});
  auto n_scan_paths_for_gen_bw_t = TPack<Int, 1, D>::zeros({n_gens_total + 1});
  // auto temp_n_nodes_for_gen_t = TPack<Int, 1, D>::zeros({n_gens_total + 1});
  // auto temp_n_scan_paths_for_gen_t =
  //     TPack<Int, 1, D>::zeros({n_gens_total + 1});
  // auto temp_n_nodes_for_gen_bw_t = TPack<Int, 1, D>::zeros({n_gens_total +
  // 1}); auto temp_n_scan_paths_for_gen_bw_t =
  //     TPack<Int, 1, D>::zeros({n_gens_total + 1});

  auto n_atoms_for_scan_path_seg_for_gen =
      n_atoms_for_scan_path_seg_for_gen_t.view;
  auto n_atoms_for_scan_path_seg_for_gen_bw =
      n_atoms_for_scan_path_seg_for_gen_bw_t.view;
  auto n_scan_paths_for_gen = n_scan_paths_for_gen_t.view;
  auto n_scan_paths_for_gen_bw = n_scan_paths_for_gen_bw_t.view;
  // auto temp_n_nodes_for_gen = temp_n_nodes_for_gen_t.view;
  // auto temp_n_scan_paths_for_gen = temp_n_scan_paths_for_gen_t.view;
  // auto temp_n_nodes_for_gen_bw = temp_n_nodes_for_gen_bw_t.view;
  // auto temp_n_scan_paths_for_gen_bw = temp_n_scan_paths_for_gen_bw_t.view;

  // printf(
  //     "size of n_atoms_for_scan_path_seg_for_gen %d: ( %d x %d)\n",
  //     n_atoms_for_scan_path_seg_for_gen.size(0),
  //     n_blocks_building_edges_total,
  //     max_n_scan_path_segs_per_gen);

  auto collect_n_atoms_for_scan_path_segs = ([=] TMOL_DEVICE_FUNC(int ind) {
    int i = ind;
    int const pose =
        i / (max_n_blocks * max_n_gens_per_bt * max_n_scan_path_segs_per_gen);
    i = i
        - pose * max_n_blocks * max_n_gens_per_bt
              * max_n_scan_path_segs_per_gen;
    int const block = i / (max_n_gens_per_bt * max_n_scan_path_segs_per_gen);
    i = i - block * max_n_gens_per_bt * max_n_scan_path_segs_per_gen;
    int const gen = i / max_n_scan_path_segs_per_gen;
    int const scan_path_seg = i % max_n_scan_path_segs_per_gen;

    // printf(
    //     "collect_n_atoms_for_scan_path_segs %d %d %d %d %d\n",
    //     ind,
    //     pose,
    //     block,
    //     gen,
    //     scan_path_seg);
    int const block_type = pose_stack_block_type[pose][block];
    if (block_type == -1) {
      return;
    }

    // During the (gen 0, scan-path-seg 0) iteration, record the number of atoms
    // for this block -- IS THIS REALLY NEEDED??? WE ALREADY HAVE
    // atom_ind_2_kfo_index if (gen == 0 && scan_path == 0) {
    //   int const block_n_atoms = block_type_n_atoms[block_type];
    // }

    int const input_conn = pose_stack_block_in_and_first_out[pose][block][0];
    int const first_out_conn =
        pose_stack_block_in_and_first_out[pose][block][1];
    if (scan_path_seg >= block_type_n_scan_path_segs[block_type][input_conn]
                                                    [first_out_conn][gen]) {
      // printf("collect_n_atoms_for_scan_path_segs early exit %d vs %d \n",
      // scan_path_seg,
      // block_type_n_scan_path_segs[block_type][input_conn][first_out_conn][gen]);
      return;
    }

    // bool const sps_is_inter_block = (gen == 0 && scan_path_seg == 0);
    // Note again: "scan path" -- a contiguous, possibly-multi-block stretch of
    // atoms to be updated together vs "scan path segment" the portion of a scan
    // path belonging to a single block. Some scan path segments are scan paths;
    // ie. they start and stop within the same block.
    bool is_root_of_scan_path = false;
    bool is_root_of_scan_path_bw = false;
    // printf(
    //     "scan path seg is interblock %d %d %d %d %d ? %d\n",
    //     block_type,
    //     input_conn,
    //     first_out_conn,
    //     gen,
    //     scan_path_seg,
    //     sps_is_inter_block);
    if (gen != 0 || scan_path_seg != 0) {
      // printf(
      //     "sps is not interblock p %d b %d g %d sps %d\n",
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg);
      is_root_of_scan_path = true;
      is_root_of_scan_path_bw = true;  // TENTATIVE!
    }

    int ff_edge_on_pose = first_ff_edge_for_block[pose][block];
    // printf("ff_edge_on_pose %d\n", ff_edge_on_pose);
    int ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;
    // note: the delay must be set based on the first FF edge for block;
    // even if this scan path segment is the root of another FF edge, we keep
    // the delay of the first FF edge for the block because the delay .
    int const ff_edge_delay = delay_for_edge[pose][ff_edge_on_pose];

    int const nj_ff_edge_rooted_at_scan_path_seg =
        non_jump_ff_edge_rooted_at_scan_path_seg[pose][block][gen]
                                                [scan_path_seg];
    int const nj_ff_edge_rooted_at_scan_path_seg_bw =
        non_jump_ff_edge_rooted_at_scan_path_seg_bw[pose][block][gen]
                                                   [scan_path_seg];
    int extra_atom_count = 0;
    bool is_root_path = false;
    if (nj_ff_edge_rooted_at_scan_path_seg != -1) {
      // printf(
      //     "nj_ff_edge_rooted_at_scan_path_seg %d %d %d %d: %d\n",
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg,
      //     nj_ff_edge_rooted_at_scan_path_seg);

      ff_edge_on_pose = nj_ff_edge_rooted_at_scan_path_seg;
      ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;
      if (is_ff_edge_root_of_scan_path[pose][ff_edge_on_pose]) {
        // printf(
        //     "forward is_root_of_scan_path: is_ff_edge_root_of_scan_path %d %d
        //     "
        //     "%d %d %d\n",
        //     pose,
        //     block,
        //     gen,
        //     scan_path_seg,
        //     ff_edge_on_pose);
        is_root_of_scan_path = true;
      }
      if (!is_edge_end_block_scan_path_seg_root_of_bw_scan_path
              [pose][block][gen][scan_path_seg]) {
        // printf(
        //     "Marking sps %d %d %d %d as not root of bw scan path\n",
        //     pose,
        //     block,
        //     gen,
        //     scan_path_seg);
        is_root_of_scan_path_bw = false;
      }
      if (is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose]) {
        // printf(
        //     "Extra atom for scan path seg %d %d %d %d; line 2346\n",
        //     pose,
        //     block,
        //     gen,
        //     scan_path_seg);
        // The scan path leaving the root of the fold forest (atom 0)
        // requires an extra atom that will not be listed in the
        // block-type's-scan path, so we add it here.
        is_root_path = true;
        extra_atom_count = 1;
      }
    }
    if (nj_ff_edge_rooted_at_scan_path_seg_bw != -1) {
      // printf(
      //     "nj_ff_edge_rooted_at_scan_path_seg_bw %d vs ff_edge_on_pose %d\n",
      //     nj_ff_edge_rooted_at_scan_path_seg_bw,
      //     ff_edge_on_pose);
      assert(ff_edge_on_pose == nj_ff_edge_rooted_at_scan_path_seg_bw);
      assert(
          ff_edge_global_index == ff_edge_on_pose + pose * max_n_edges_per_ff);
      if (is_ff_edge_root_of_scan_path_bw[pose][ff_edge_on_pose]) {
        // printf(
        //     "backward is_root_of_scan_path_bw:
        //     is_ff_edge_root_of_scan_path_bw "
        //     "%d %d %d %d %d\n",
        //     pose,
        //     block,
        //     gen,
        //     scan_path_seg,
        //     ff_edge_on_pose);
        is_root_of_scan_path_bw = true;
      }
    }

    int const ff_edge_type = ff_edges[pose][ff_edge_on_pose][0];
    // printf(
    //     "pose %d block %d gen %d scan_path_seg %d, ff_edge_on_pose %d, "
    //     "ff_edge_type %d\n",
    //     pose,
    //     block,
    //     gen,
    //     scan_path_seg,
    //     ff_edge_on_pose,
    //     ff_edge_type);
    if (ff_edge_type == 1) {
      int const j_ff_edge_rooted_at_scan_path_seg =
          jump_ff_edge_rooted_at_scan_path_seg[pose][block][gen][scan_path_seg];
      // printf(
      //     "j_ff_edge_rooted_at_scan_path_seg[%d][%d][%d][%d] = %d\n",
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg,
      //     j_ff_edge_rooted_at_scan_path_seg);
      if (j_ff_edge_rooted_at_scan_path_seg != -1) {
        ff_edge_on_pose = j_ff_edge_rooted_at_scan_path_seg;
        ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;

        is_root_path = is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose];
        is_root_of_scan_path = true;  // Is this always true???
        if (is_ff_edge_root_of_scan_path[pose][ff_edge_on_pose]) {
          // printf(
          //     "Extra atom for scan path seg %d %d %d %d; line 2399\n",
          //     pose,
          //     block,
          //     gen,
          //     scan_path_seg);
          // Jump edge that's rooted at this scan path. For this
          // edge we must add an extra atom representing the
          // start-block atom: it will not be listed as one
          // of the atoms in the block-type's-scan path. This works
          // both for jump edges in the middle of a fold tree as
          // well as for the jump edge that connects the root of the
          // fold forest (atom 0) to the root of the fold tree for
          // this Pose; in the latter case, the start block for the
          // jump is considered the block that roots the scan path
          // seg, rather than non-root-jump edges that consider the
          // end block as rooting the scan path seg, so the atom
          // on the start block will already be accounted for.
          extra_atom_count = 1;
        }
      }
      int const j_ff_edge_rooted_at_scan_path_seg_bw =
          jump_ff_edge_rooted_at_scan_path_seg_bw[pose][block][gen]
                                                 [scan_path_seg];
      if (j_ff_edge_rooted_at_scan_path_seg_bw != -1) {
        assert(ff_edge_on_pose == j_ff_edge_rooted_at_scan_path_seg_bw);
        assert(
            ff_edge_global_index
            == ff_edge_on_pose + pose * max_n_edges_per_ff);
        if (is_ff_edge_root_of_scan_path_bw[pose][ff_edge_on_pose]) {
          // printf(
          //     "is_ff_edge_root_of_scan_path_bw %d %d %d %d %d\n",
          //     pose,
          //     block,
          //     gen,
          //     scan_path_seg,
          //     ff_edge_on_pose);
          is_root_of_scan_path_bw = true;
        }
      }
    }

    // printf("ff_edge_global_index %d\n", ff_edge_global_index);
    // printf("ff_edge_delay %d\n", ff_edge_delay);
    int const ff_edge_gen = gen + ff_edge_delay;
    int const ff_edge_gen_bw = (n_gens_total - 1) - ff_edge_gen;
    // printf("ff_edge_gen %d\n", ff_edge_gen);
    int block_position_on_ff_edge = 0;
    int block_position_on_ff_edge_bw = 0;
    if (ff_edge_type == 1) {
      // Jump edge -- the start block is block position 0, the end block is
      // block position 1.
      block_position_on_ff_edge =
          (block == ff_edges[pose][ff_edge_on_pose][1] ? 0 : 1);
      block_position_on_ff_edge_bw =
          (block == ff_edges[pose][ff_edge_on_pose][1] ? 1 : 0);
    } else {
      auto fw_and_bw_block_positions =
          polymer_edge_index_for_block(ff_edges, pose, ff_edge_on_pose, block);
      block_position_on_ff_edge = std::get<0>(fw_and_bw_block_positions);
      block_position_on_ff_edge_bw = std::get<1>(fw_and_bw_block_positions);
    }
    // printf(
    //     "block_position_on_ff_edge %d (%d, %d-> %d), "
    //     "block_position_on_ff_edge_bw %d\n",
    //     block_position_on_ff_edge,
    //     block,
    //     ff_edges[pose][ff_edge_on_pose][1],
    //     ff_edges[pose][ff_edge_on_pose][2],
    //     block_position_on_ff_edge_bw);

    int const edge_toposort_index =
        topo_sort_index_for_edge[ff_edge_global_index];
    int const edge_toposort_index_bw =
        n_poses * max_n_edges_per_ff - 1 - edge_toposort_index;

    int boftsfg = block_offset_for_tsedge_for_gen
        [ff_edge_gen * n_poses * max_n_edges_per_ff + edge_toposort_index];
    int boftsfg_bw = block_offset_for_tsedge_for_gen_bw
        [ff_edge_gen_bw * n_poses * max_n_edges_per_ff
         + edge_toposort_index_bw];
    // printf("boftsfg %d boftsfg_bw %d\n", boftsfg, boftsfg_bw);

    int sps_index_in_n_atoms_offset =
        (block_position_on_ff_edge + boftsfg) * max_n_scan_path_segs_per_gen
        + scan_path_seg;
    int sps_index_in_n_atoms_offset_bw =
        (block_position_on_ff_edge_bw + boftsfg_bw)
            * max_n_scan_path_segs_per_gen
        + scan_path_seg;
    int n_atoms_for_scan_path_seg =
        block_type_scan_path_seg_length[block_type][input_conn][first_out_conn]
                                       [gen][scan_path_seg];
    // printf(
    //     "sps_index_in_n_atoms_offset %d = (%d + %d) * %d + %d; "
    //     "sps_index_in_n_atoms_offset_bw %d = (%d + %d) * %d + %d\n",
    //     sps_index_in_n_atoms_offset,
    //     block_position_on_ff_edge,
    //     boftsfg,
    //     max_n_scan_path_segs_per_gen,
    //     scan_path_seg,
    //     sps_index_in_n_atoms_offset_bw,
    //     block_position_on_ff_edge_bw,
    //     boftsfg_bw,
    //     max_n_scan_path_segs_per_gen,
    //     scan_path_seg);

    // printf(
    //     "p %d b %d g %d sp %d e %d (%d: %d->%d), ffeg %d, bo4ts4g %d, spio %d
    //     " "nats %d+%d\n", pose, block, gen, scan_path, ff_edge_on_pose,
    //     ff_edge_type,
    //     ff_edges[pose][ff_edge_on_pose][1],
    //     ff_edges[pose][ff_edge_on_pose][2],
    //     ff_edge_gen,
    //     block_offset_for_tsedge_for_gen
    //         [ff_edge_gen * n_poses * max_n_edges_per_ff +
    //         edge_toposort_index],
    //     sp_index_in_n_atoms_offset,
    //     n_atoms_for_scan_path,
    //     extra_atom_count);

    n_atoms_for_scan_path_seg_for_gen[sps_index_in_n_atoms_offset] =
        n_atoms_for_scan_path_seg + extra_atom_count;  // ...TADA!

    n_atoms_for_scan_path_seg_for_gen_bw[sps_index_in_n_atoms_offset_bw] =
        n_atoms_for_scan_path_seg + extra_atom_count;

    if (is_root_of_scan_path) {
      // printf(
      //     "is_root_of_scan_path fw: %d (p %d b %d g %d sps %d)\n",
      //     sps_index_in_n_atoms_offset,
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg);
      is_scan_path_seg_root_of_scan_path[sps_index_in_n_atoms_offset] = 1;
      accumulate<D, Int>::add(n_scan_paths_for_gen[ff_edge_gen], 1);
    }
    if (is_root_of_scan_path_bw) {
      // printf(
      //     "is_root_of_scan_path bw: %d (p %d b %d g %d sps %d)\n",
      //     sps_index_in_n_atoms_offset_bw,
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg);
      is_scan_path_seg_root_of_scan_path_bw[sps_index_in_n_atoms_offset_bw] = 1;
      accumulate<D, Int>::add(n_scan_paths_for_gen_bw[ff_edge_gen_bw], 1);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks * max_n_gens_per_bt * max_n_scan_path_segs_per_gen,
      collect_n_atoms_for_scan_path_segs);
  gpuErrPeek;
  gpuErrSync;

  // Step 12
  // Step N-1:
  // And with the number of atoms for each scan path segment, we can now
  // calculate their offsets in the nodes tensor using scan
  // printf("Step 12\n");
  auto nodes_offset_for_scan_path_seg_for_gen_tp = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto nodes_offset_for_scan_path_seg_for_gen_bw_tp = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto root_scan_path_offset_tp = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto root_scan_path_offset_bw_tp = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto n_scan_path_offsets_for_gen_t =
      TPack<Int, 1, D>::zeros({n_gens_total + 1});
  auto n_scan_path_offsets_for_gen_bw_t =
      TPack<Int, 1, D>::zeros({n_gens_total + 1});
  // auto temp_nodes_offset_for_gen_t =
  //     TPack<Int, 1, D>::zeros({n_gens_total + 1});
  // auto temp_nodes_offset_for_gen_bw_t =
  //     TPack<Int, 1, D>::zeros({n_gens_total + 1});

  auto nodes_offset_for_scan_path_seg_for_gen =
      nodes_offset_for_scan_path_seg_for_gen_tp.view;
  auto nodes_offset_for_scan_path_seg_for_gen_bw =
      nodes_offset_for_scan_path_seg_for_gen_bw_tp.view;
  auto root_scan_path_offset = root_scan_path_offset_tp.view;
  auto root_scan_path_offset_bw = root_scan_path_offset_bw_tp.view;
  auto n_scan_path_offsets_for_gen = n_scan_path_offsets_for_gen_t.view;
  auto n_scan_path_offsets_for_gen_bw = n_scan_path_offsets_for_gen_bw_t.view;
  // auto temp_nodes_offset_for_gen = temp_nodes_offset_for_gen_t.view;
  // auto temp_nodes_offset_for_gen_bw = temp_nodes_offset_for_gen_bw_t.view;

  int n_nodes_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_atoms_for_scan_path_seg_for_gen.data(),
          nodes_offset_for_scan_path_seg_for_gen.data(),
          n_blocks_building_edges_total * max_n_scan_path_segs_per_gen,
          mgpu::plus_t<Int>());
  int n_nodes_total2 =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_atoms_for_scan_path_seg_for_gen_bw.data(),
          nodes_offset_for_scan_path_seg_for_gen_bw.data(),
          n_blocks_building_edges_total * max_n_scan_path_segs_per_gen,
          mgpu::plus_t<Int>());
  int n_scan_path_roots_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          is_scan_path_seg_root_of_scan_path.data(),
          root_scan_path_offset.data(),
          n_blocks_building_edges_total * max_n_scan_path_segs_per_gen,
          mgpu::plus_t<Int>());
  int n_scan_path_roots_total2 =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          is_scan_path_seg_root_of_scan_path_bw.data(),
          root_scan_path_offset_bw.data(),
          n_blocks_building_edges_total * max_n_scan_path_segs_per_gen,
          mgpu::plus_t<Int>());
  // printf(
  //     "n_scan_path_roots_total (fw) %d\n n_scan_path_roots_total2 (bw):
  //     %d\n", n_scan_path_roots_total, n_scan_path_roots_total2);

  DeviceDispatch<D>::template scan<mgpu::scan_type_exc>(
      n_scan_paths_for_gen.data(),
      n_scan_path_offsets_for_gen.data(),
      n_gens_total + 1,
      mgpu::plus_t<Int>());
  DeviceDispatch<D>::template scan<mgpu::scan_type_exc>(
      n_scan_paths_for_gen_bw.data(),
      n_scan_path_offsets_for_gen_bw.data(),
      n_gens_total + 1,
      mgpu::plus_t<Int>());
  // DeviceDispatch<D>::template scan<mgpu::scan_type_exc>(
  //     temp_n_nodes_for_gen.data(),
  //     temp_nodes_offset_for_gen.data(),
  //     n_gens_total + 1,
  //     mgpu::plus_t<Int>());
  // DeviceDispatch<D>::template scan<mgpu::scan_type_exc>(
  //     temp_n_nodes_for_gen_bw.data(),
  //     temp_nodes_offset_for_gen_bw.data(),
  //     n_gens_total + 1,
  //     mgpu::plus_t<Int>());

  // for (int gen = 0; gen < n_gens_total + 1; ++gen) {
  //   int const gen_bw = n_gens_total - gen;
  //   int const tsedge0_block_offset =
  //       gen < n_gens_total ? block_offset_for_tsedge_for_gen
  //               [gen * n_poses * max_n_edges_per_ff]
  //                          : n_blocks_building_edges_total;
  //   int const tsedge0_block_offset_bw =
  //       gen_bw < n_gens_total ? block_offset_for_tsedge_for_gen_bw
  //               [gen_bw * n_poses * max_n_edges_per_ff]
  //                             : n_blocks_building_edges_total;
  //   // printf(
  //   //     "tsedge0 for gen index %d * %d * %d = %d, and offset = %d\n",
  //   //     gen,
  //   //     n_poses,
  //   //     max_n_edges_per_ff,
  //   //     gen * n_poses * max_n_edges_per_ff,
  //   //     tsedge0_block_offset);
  //   // n_gens_x_n_edges;
  //   int const tsedge0_for_gen =
  //       tsedge0_block_offset < n_blocks_building_edges_total
  //           ? tsedge0_block_offset * max_n_scan_path_segs_per_gen
  //           : -1;
  //   int const tsedge0_for_gen_bw =
  //       tsedge0_block_offset_bw < n_blocks_building_edges_total
  //           ? tsedge0_block_offset_bw * max_n_scan_path_segs_per_gen
  //           : -1;
  //   int const tsedge0_node_offset =
  //       gen < n_gens_total
  //               && tsedge0_block_offset < n_blocks_building_edges_total
  //           ? nodes_offset_for_scan_path_seg_for_gen[tsedge0_for_gen]
  //           : n_nodes_total;
  //   int const tsedge0_node_offset_bw =
  //       gen_bw < n_gens_total
  //               && tsedge0_block_offset_bw < n_blocks_building_edges_total
  //           ? nodes_offset_for_scan_path_seg_for_gen_bw[tsedge0_for_gen_bw]
  //           : n_nodes_total;
  //   int const tsedge0_root_offset =
  //       gen < n_gens_total
  //               && tsedge0_block_offset < n_blocks_building_edges_total
  //           ? root_scan_path_offset[tsedge0_for_gen]
  //           : n_scan_path_roots_total;
  //   int const tsedge0_root_offset_bw =
  //       gen_bw < n_gens_total
  //               && tsedge0_block_offset_bw < n_blocks_building_edges_total
  //           ? root_scan_path_offset_bw[tsedge0_for_gen_bw]
  //           : n_scan_path_roots_total;
  //   // printf(
  //   //     "gen %d n_scan_paths %d n_nodes %d sp_offset %d nodes offset %d
  //   //     tsedg0 "
  //   //     "%d %d\n",
  //   //     gen,
  //   //     n_scan_paths_for_gen[gen],
  //   //     temp_n_nodes_for_gen[gen],
  //   //     n_scan_path_offsets_for_gen[gen],
  //   //     temp_nodes_offset_for_gen[gen],
  //   //     tsedge0_node_offset,
  //   //     tsedge0_root_offset);
  //   // printf(
  //   //     "gen_bw %d n_scan_paths %d n_nodes %d sp_offset %d nodes offset %d
  //   "
  //   //     "tsedg0 %d %d\n",
  //   //     gen_bw,
  //   //     n_scan_paths_for_gen_bw[gen_bw],
  //   //     temp_n_nodes_for_gen[gen_bw],
  //   //     n_scan_path_offsets_for_gen_bw[gen_bw],
  //   //     temp_nodes_offset_for_gen_bw[gen],
  //   //     tsedge0_node_offset_bw,
  //   //     tsedge0_root_offset_bw);
  // }

  // for (int ind = 0;
  //      ind < n_blocks_building_edges_total * max_n_scan_paths_per_gen;
  //      ++ind) {
  //   int i = ind;
  //   printf(
  //       "nodes_offset_for_scan_path_for_gen[%d] = %d\n",
  //       ind,
  //       nodes_offset_for_scan_path_for_gen[ind]);
  // }
  // printf("n_scan_path_roots_total %d\n", n_scan_path_roots_total);
  // for (int ind = 0;
  //      ind < n_blocks_building_edges_total * max_n_scan_paths_per_gen;
  //      ++ind) {
  //   int i = ind;
  //   printf(
  //       "root_scan_path_offset[%d] = %d\n",
  //       ind,
  //       root_scan_path_offset[ind]);
  // }

  // Step 13
  // Step N:
  // And we can now, finally, copy the scan-path-segment stencils into
  // the nodes tensor
  // printf(
  //     "Step 13, n_nodes_total %d n_scan_path_roots_total %d\n",
  //     n_nodes_total,
  //     n_scan_path_roots_total);
  // Fill both the forward- and backward paths at the same time.
  auto nodes_fw_t = TPack<Int, 1, D>::full(n_nodes_total, -1);
  auto nodes_fw = nodes_fw_t.view;
  auto scans_fw_t = TPack<Int, 1, D>::full({n_scan_path_roots_total}, -1);
  auto scans_fw = scans_fw_t.view;
  auto gens_fw_t = TPack<Int, 2, D>::full({n_gens_total + 1, 2}, -1);
  auto gens_fw = gens_fw_t.view;

  auto nodes_bw_t = TPack<Int, 1, D>::full(n_nodes_total, -1);
  auto nodes_bw = nodes_bw_t.view;
  auto scans_bw_t = TPack<Int, 1, D>::full({n_scan_path_roots_total}, -1);
  auto scans_bw = scans_bw_t.view;
  auto gens_bw_t = TPack<Int, 2, D>::full({n_gens_total + 1, 2}, -1);
  auto gens_bw = gens_bw_t.view;

  auto n_scans_per_gen_t = TPack<Int, 1, D>::full({n_gens_total}, 0);
  auto n_nodes_per_gen_t = TPack<Int, 1, D>::full({n_gens_total}, 0);
  auto n_scans_per_gen = n_scans_per_gen_t.view;
  auto n_nodes_per_gen = n_nodes_per_gen_t.view;

  // auto fill_gens_offset_tensors = ([=] TMOL_DEVICE_FUNC(int ind) {
  //   int const gen_bw = n_gens_total - ind;
  //   int const tsedge0_block_offset =
  //       ind < n_gens_total ? block_offset_for_tsedge_for_gen
  //                                [ind * n_poses * max_n_edges_per_ff]
  //                          : n_blocks_building_edges_total;
  //   int const tsedge0_block_offset_bw =
  //       gen_bw < n_gens_total ? block_offset_for_tsedge_for_gen_bw
  //                                   [gen_bw * n_poses * max_n_edges_per_ff]
  //                             : n_blocks_building_edges_total;
  //   int const tsedge0_for_gen =
  //       tsedge0_block_offset < n_blocks_building_edges_total
  //           ? tsedge0_block_offset * max_n_scan_path_segs_per_gen
  //           : -1;
  //   int const tsedge0_for_gen_bw =
  //       tsedge0_block_offset_bw < n_blocks_building_edges_total
  //           ? tsedge0_block_offset_bw * max_n_scan_path_segs_per_gen
  //           : -1;
  //   int const tsedge0_node_offset =
  //       ind < n_gens_total
  //               && tsedge0_block_offset < n_blocks_building_edges_total
  //           ? nodes_offset_for_scan_path_seg_for_gen[tsedge0_for_gen]
  //           : n_nodes_total;
  //   int const tsedge0_node_offset_bw =
  //       gen_bw < n_gens_total
  //               && tsedge0_block_offset_bw < n_blocks_building_edges_total
  //           ? nodes_offset_for_scan_path_seg_for_gen_bw[tsedge0_for_gen_bw]
  //           : n_nodes_total;
  //   int const tsedge0_root_offset =
  //       ind < n_gens_total
  //               && tsedge0_block_offset < n_blocks_building_edges_total
  //           ? root_scan_path_offset[tsedge0_for_gen]
  //           : n_scan_path_roots_total;
  //   int const tsedge0_root_offset_bw =
  //       gen_bw < n_gens_total
  //               && tsedge0_block_offset_bw < n_blocks_building_edges_total
  //           ? root_scan_path_offset_bw[tsedge0_for_gen_bw]
  //           : n_scan_path_roots_total;

  //   gens_fw[ind][0] = tsedge0_node_offset;
  //   gens_fw[ind][1] = tsedge0_root_offset;
  //   gens_bw[gen_bw][0] = tsedge0_node_offset_bw;
  //   gens_bw[gen_bw][1] = tsedge0_root_offset_bw;
  //   //   printf(
  //   //       "gens_fw[%d][:] = (%d, %d); gens_bw[%d][:] = (%d, %d)\n",
  //   //       ind,
  //   //       gens_fw[ind][0],
  //   //       gens_fw[ind][1],
  //   //       gen_bw,
  //   //       gens_bw[gen_bw][0],
  //   //       gens_bw[gen_bw][1]);
  // });
  // DeviceDispatch<D>::template forall<launch_t>(
  //     n_gens_total + 1, fill_gens_offset_tensors);

  auto fill_nodes_tensor_from_scan_path_seg_stencils = ([=] TMOL_DEVICE_FUNC(
                                                            int ind) {
    int i = ind;
    int const pose =
        i / (max_n_blocks * max_n_gens_per_bt * max_n_scan_path_segs_per_gen);
    i = i
        - pose * max_n_blocks * max_n_gens_per_bt
              * max_n_scan_path_segs_per_gen;
    int const block = i / (max_n_gens_per_bt * max_n_scan_path_segs_per_gen);
    i = i - block * max_n_gens_per_bt * max_n_scan_path_segs_per_gen;
    int const gen = i / max_n_scan_path_segs_per_gen;
    int const scan_path_seg = i % max_n_scan_path_segs_per_gen;

    if (ind <= n_gens_total) {
      int const gen_bw = n_gens_total - ind;
      int const tsedge0_block_offset =
          ind < n_gens_total ? block_offset_for_tsedge_for_gen
                  [ind * n_poses * max_n_edges_per_ff]
                             : n_blocks_building_edges_total;
      int const tsedge0_block_offset_bw =
          gen_bw < n_gens_total ? block_offset_for_tsedge_for_gen_bw
                  [gen_bw * n_poses * max_n_edges_per_ff]
                                : n_blocks_building_edges_total;
      int const tsedge0_for_gen =
          tsedge0_block_offset < n_blocks_building_edges_total
              ? tsedge0_block_offset * max_n_scan_path_segs_per_gen
              : -1;
      int const tsedge0_for_gen_bw =
          tsedge0_block_offset_bw < n_blocks_building_edges_total
              ? tsedge0_block_offset_bw * max_n_scan_path_segs_per_gen
              : -1;
      int const tsedge0_node_offset =
          ind < n_gens_total
                  && tsedge0_block_offset < n_blocks_building_edges_total
              ? nodes_offset_for_scan_path_seg_for_gen[tsedge0_for_gen]
              : n_nodes_total;
      int const tsedge0_node_offset_bw =
          gen_bw < n_gens_total
                  && tsedge0_block_offset_bw < n_blocks_building_edges_total
              ? nodes_offset_for_scan_path_seg_for_gen_bw[tsedge0_for_gen_bw]
              : n_nodes_total;
      int const tsedge0_root_offset =
          ind < n_gens_total
                  && tsedge0_block_offset < n_blocks_building_edges_total
              ? root_scan_path_offset[tsedge0_for_gen]
              : n_scan_path_roots_total;
      int const tsedge0_root_offset_bw =
          gen_bw < n_gens_total
                  && tsedge0_block_offset_bw < n_blocks_building_edges_total
              ? root_scan_path_offset_bw[tsedge0_for_gen_bw]
              : n_scan_path_roots_total;

      gens_fw[ind][0] = tsedge0_node_offset;
      gens_fw[ind][1] = tsedge0_root_offset;
      gens_bw[gen_bw][0] = tsedge0_node_offset_bw;
      gens_bw[gen_bw][1] = tsedge0_root_offset_bw;
      //   printf(
      //       "gens_fw[%d][:] = (%d, %d); gens_bw[%d][:] = (%d, %d)\n",
      //       ind,
      //       gens_fw[ind][0],
      //       gens_fw[ind][1],
      //       gen_bw,
      //       gens_bw[gen_bw][0],
      //       gens_bw[gen_bw][1]);
    }

    if (pose >= n_poses) {
      // it is possible, though unlikely, that the max(n_segments, n_gens_total
      // + 1) where n_segments = n_poses * max_n_blocks * max_n_gens_per_bt *
      // max_n_scan_path_segs_per_gen is n_gens_total+1, and so we must check
      // that this thread index is in bounds before proceeding.
      return;
    }
    // printf(
    //     "fill_nodes_tensor_from_scan_path_seg_stencils %d %d %d %d %d\n",
    //     i,
    //     pose,
    //     block,
    //     gen,
    //     scan_path_seg);

    int const block_type = pose_stack_block_type[pose][block];
    if (block_type == -1) {
      return;
    }
    int const input_conn = pose_stack_block_in_and_first_out[pose][block][0];
    int const first_out_conn =
        pose_stack_block_in_and_first_out[pose][block][1];
    assert(input_conn >= 0 && input_conn < max_n_input_conn + 2);
    assert(first_out_conn >= 0 && first_out_conn <= max_n_output_conn + 1);
    if (scan_path_seg >= block_type_n_scan_path_segs[block_type][input_conn]
                                                    [first_out_conn][gen]) {
      // printf("collect_n_atoms_for_scan_paths early exit %d vs %d \n",
      // scan_path,
      // block_type_n_scan_paths[block_type][input_conn][first_out_conn][gen]);
      return;
    }

    // printf("1\n");
    bool is_edge_ft_root = false;
    bool is_bt_scan_path_seg_root_of_own_scan_path = false;
    int ff_edge_on_pose = first_ff_edge_for_block[pose][block];
    int ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;
    // note: this must be set based on the first FF edge for block;
    // even if this scan path is the root of another FF edge, we keep
    // the delay of the first FF edge for the block.
    int const ff_edge_delay = delay_for_edge[pose][ff_edge_on_pose];
    int const nj_ff_edge_rooted_at_scan_path_seg =
        non_jump_ff_edge_rooted_at_scan_path_seg[pose][block][gen]
                                                [scan_path_seg];

    int extra_atom_count = 0;
    if (nj_ff_edge_rooted_at_scan_path_seg != -1) {
      // printf("nj_ff_edge_rooted_at_scan_path %d\n",
      // nj_ff_edge_rooted_at_scan_path);
      ff_edge_on_pose = nj_ff_edge_rooted_at_scan_path_seg;
      ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;
      is_edge_ft_root = is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose];
      if (is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose]) {
        // printf(
        //     "Extra atom for scan path seg %d %d %d %d; line 2912\n",
        //     pose,
        //     block,
        //     gen,
        //     scan_path_seg);
        // The path leaving the root of the fold forest (atom 0)
        // requires an extra atom that will not be listed in the
        // block-type's-scan path, so we add it here.
        // printf(
        //     "is_ff_edge_root_of_fold_tree p %d b %d g %d sps %d, first edge "
        //     "%d, nj_edge %d\n",
        //     pose,
        //     block,
        //     gen,
        //     scan_path_seg,
        //     first_ff_edge_for_block[pose][block],
        //     nj_ff_edge_rooted_at_scan_path_seg);
        extra_atom_count = 1;
      }
    }
    int const ff_edge_type = ff_edges[pose][ff_edge_on_pose][0];
    if (ff_edge_type == 1) {
      int const j_ff_edge_rooted_at_scan_path_seg =
          jump_ff_edge_rooted_at_scan_path_seg[pose][block][gen][scan_path_seg];
      // printf(
      //     "jump_ff_edge_rooted_at_scan_path_seg[%d][%d][%d][%d] = %d\n",
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg,
      //     j_ff_edge_rooted_at_scan_path_seg);
      if (j_ff_edge_rooted_at_scan_path_seg != -1) {
        // bool const block_is_first = block ==
        // ff_edges[pose][ff_edge_on_pose][1];
        is_edge_ft_root = is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose];
        if (is_ff_edge_root_of_scan_path[pose][ff_edge_on_pose]) {
          // printf(
          //     "Extra atom for scan path seg %d %d %d %d; line 2928\n",
          //     pose,
          //     block,
          //     gen,
          //     scan_path_seg);
          // Jump edge that's rooted at this scan path. For this
          // edge we must add an extra atom representing the
          // start-block atom: it will not be listed as one
          // of the atoms in the block-type's-scan path. This works
          // both for jump edges in the middle of a fold tree as
          // well as for the jump edge that connects the root of the
          // fold forest (atom 0) to the root of the fold tree for
          // this Pose.
          // printf(
          //     "is_ff_edge_root_of_fold_tree p %d b %d g %d sps %d, first edge
          //     "
          //     "%d, j_edge %d\n",
          //     pose,
          //     block,
          //     gen,
          //     scan_path_seg,
          //     first_ff_edge_for_block[pose][block],
          //     j_ff_edge_rooted_at_scan_path_seg);
          extra_atom_count = 1;
        }
      }
    }
    // printf("2\n");
    // printf("ff_edge_global_index %d\n", ff_edge_global_index);
    // printf("ff_edge_delay %d\n", ff_edge_delay);
    // int const ff_edge_type = ff_edges[pose][ff_edge_on_pose][0];
    int const ff_edge_gen = gen + ff_edge_delay;
    int const ff_edge_gen_bw = (n_gens_total - 1) - ff_edge_gen;
    // printf("ff_edge_gen %d\n", ff_edge_gen);
    int block_position_on_ff_edge = 0;
    int block_position_on_ff_edge_bw = 0;
    if (ff_edge_type == 1) {
      // Jump edge -- the start block is block position 0, the end block is
      // block position 1.
      block_position_on_ff_edge =
          (block == ff_edges[pose][ff_edge_on_pose][1] ? 0 : 1);
      block_position_on_ff_edge_bw =
          (block == ff_edges[pose][ff_edge_on_pose][1] ? 1 : 0);
    } else {
      auto fw_and_bw_block_positions =
          polymer_edge_index_for_block(ff_edges, pose, ff_edge_on_pose, block);
      block_position_on_ff_edge = std::get<0>(fw_and_bw_block_positions);
      block_position_on_ff_edge_bw = std::get<1>(fw_and_bw_block_positions);
      // block_position_on_ff_edge =
      //     polymer_edge_index_for_block(ff_edges, pose, ff_edge_on_pose,
      //     block);
    }
    // printf("block_position_on_ff_edge %d\n", block_position_on_ff_edge);

    int edge_toposort_index = topo_sort_index_for_edge[ff_edge_global_index];
    int const edge_toposort_index_bw =
        n_poses * max_n_edges_per_ff - 1 - edge_toposort_index;
    // printf(
    //     "edge_toposort_index %d edge_toposort_index_bw %d\n",
    //     edge_toposort_index,
    //     edge_toposort_index_bw);
    int boftsfg = block_offset_for_tsedge_for_gen
        [ff_edge_gen * n_poses * max_n_edges_per_ff + edge_toposort_index];
    int boftsfg_bw = block_offset_for_tsedge_for_gen_bw
        [ff_edge_gen_bw * n_poses * max_n_edges_per_ff
         + edge_toposort_index_bw];
    // printf("boftsfg %d boftsfg_bw %d\n", boftsfg, boftsfg_bw);

    // What is the block offset for the first edge (topo-sort edge 0) for
    // this generation?
    int const tsedge0_block_offset =
        ff_edge_gen < n_gens_total ? block_offset_for_tsedge_for_gen
                [ff_edge_gen * n_poses * max_n_edges_per_ff]
                                   : n_blocks_building_edges_total;
    int const tsedge0_block_offset_bw =
        ff_edge_gen_bw < n_gens_total
            ? block_offset_for_tsedge_for_gen_bw
                [ff_edge_gen_bw * n_poses * max_n_edges_per_ff]
            : n_blocks_building_edges_total;  // What is the offset for the
                                              // first scan path segment for
                                              // tsegde0?
    int const tsedge0_for_gen =
        tsedge0_block_offset < n_blocks_building_edges_total
            ? tsedge0_block_offset * max_n_scan_path_segs_per_gen
            : -1;
    int const tsedge0_for_gen_bw =
        tsedge0_block_offset_bw < n_blocks_building_edges_total
            ? tsedge0_block_offset_bw * max_n_scan_path_segs_per_gen
            : -1;
    // What is the index of the first scan path segment in the nodes tensor?
    int const tsedge0_node_offset =
        ff_edge_gen < n_gens_total
                && tsedge0_block_offset < n_blocks_building_edges_total
            ? nodes_offset_for_scan_path_seg_for_gen[tsedge0_for_gen]
            : n_nodes_total;
    int const tsedge0_node_offset_bw =
        ff_edge_gen_bw < n_gens_total
                && tsedge0_block_offset_bw < n_blocks_building_edges_total
            ? nodes_offset_for_scan_path_seg_for_gen_bw[tsedge0_for_gen_bw]
            : n_nodes_total;
    // What is the index of the first scan path for tsegde0?
    int const tsedge0_root_offset =
        ff_edge_gen < n_gens_total
                && tsedge0_block_offset < n_blocks_building_edges_total
            ? root_scan_path_offset[tsedge0_for_gen]
            : n_scan_path_roots_total;
    int const tsedge0_root_offset_bw =
        ff_edge_gen_bw < n_gens_total
                && tsedge0_block_offset_bw < n_blocks_building_edges_total
            ? root_scan_path_offset_bw[tsedge0_for_gen_bw]
            : n_scan_path_roots_total;
    // printf(
    //     "tsedge0_block_offset %d tsedge0_for_gen %d tsedge0_node_offset %d "
    //     "tsedge0_root_offset %d\n",
    //     tsedge0_block_offset,
    //     tsedge0_for_gen,
    //     tsedge0_node_offset,
    //     tsedge0_root_offset);
    // printf(
    //     "tsedge0_block_offset_bw %d tsedge0_for_gen_bw %d "
    //     "tsedge0_node_offset_bw %d tsedge0_root_offset_bw %d\n",
    //     tsedge0_block_offset_bw,
    //     tsedge0_for_gen_bw,
    //     tsedge0_node_offset_bw,
    //     tsedge0_root_offset_bw);

    // printf(
    //     "boftsfg = block_offset_for_tsedge_for_gen[%d * %d * %d + %d] =
    //     %d\n", ff_edge_gen, n_poses, max_n_edges_per_ff, edge_toposort_index,
    //     boftsfg);
    // printf(
    //     "sp_index_in_n_atoms_offset calc: %d + %d * %d (%d) + %d * %d
    //     (%d)\n", scan_path, block_position_on_ff_edge,
    //     max_n_scan_paths_per_gen,
    //     block_position_on_ff_edge * max_n_scan_paths_per_gen,
    //     boftsfg,
    //     max_n_scan_paths_per_gen,
    //     boftsfg * max_n_scan_paths_per_gen);
    // printf("3\n");
    int sps_index_in_n_atoms_offset =
        (block_position_on_ff_edge + boftsfg) * max_n_scan_path_segs_per_gen
        + scan_path_seg;
    int sps_index_in_n_atoms_offset_bw =
        (block_position_on_ff_edge_bw + boftsfg_bw)
            * max_n_scan_path_segs_per_gen
        + scan_path_seg;
    // printf(
    //     "sp_index_in_n_atoms_offset %d = (%d + %d) * %d + %d; "
    //     "sp_index_in_n_atoms_offset_bw %d = (%d + %d) * %d + %d\n",
    //     sps_index_in_n_atoms_offset,
    //     block_position_on_ff_edge,
    //     boftsfg,
    //     max_n_scan_path_segs_per_gen,
    //     scan_path_seg,
    //     sps_index_in_n_atoms_offset_bw,
    //     block_position_on_ff_edge_bw,
    //     boftsfg_bw,
    //     max_n_scan_path_segs_per_gen,
    //     scan_path_seg);

    int const nodes_offset =
        nodes_offset_for_scan_path_seg_for_gen[sps_index_in_n_atoms_offset];
    int const nodes_offset_bw = nodes_offset_for_scan_path_seg_for_gen_bw
        [sps_index_in_n_atoms_offset_bw];
    // printf("nodes_offset_bw %d\n", nodes_offset_bw);
    // printf(
    //     "p %d b %d g %d sp %d e %d (%d: %d->%d), ffeg %d, bo4ts4g %d, spio %d
    //     " "nodes_offset %d x %d\n", pose, block, gen, scan_path,
    //     ff_edge_on_pose,
    //     ff_edge_type,
    //     ff_edges[pose][ff_edge_on_pose][1],
    //     ff_edges[pose][ff_edge_on_pose][2],
    //     ff_edge_gen,
    //     block_offset_for_tsedge_for_gen
    //         [ff_edge_gen * n_poses * max_n_edges_per_ff +
    //         edge_toposort_index],
    //     sp_index_in_n_atoms_offset,
    //     nodes_offset,
    //     extra_atom_count);
    // printf("sp_index_in_n_atoms_offset %d = %d + %d * %d +
    // block_offset_for_tsedge_for_gen[%d * %d * %d + %d] = % d * %d\n",
    //   sp_index_in_n_atoms_offset, scan_path, block_position_on_ff_edge,
    //   max_n_scan_paths_per_gen, ff_edge_delay, n_poses,  max_n_edges_per_ff,
    //   edge_toposort_index,
    //   block_offset_for_tsedge_for_gen[
    //     ff_edge_delay * n_poses * max_n_edges_per_ff +
    //     edge_toposort_index
    //   ], max_n_scan_paths_per_gen);

    // int const ff_edge_gen_topo_sort_index =
    //     ff_edge_gen * n_poses * max_n_edges_per_ff
    //     + topo_sort_index_for_edge[ff_edge_global_index];
    // printf("ff_edge_gen_topo_sort_index %d\n", ff_edge_gen_topo_sort_index);
    // int const ff_edge_gen_scan_path_offset =
    //     sp_offset_for_ff_edge_for_gen_by_topo_sort[ff_edge_gen_topo_sort_index];
    // printf("ff_edge_gen_scan_path_offset %d\n",
    // ff_edge_gen_scan_path_offset); int const n_atoms_for_scan_path_index =
    //     ff_edge_gen_scan_path_offset + block_position_on_ff_edge;
    // printf("n_atoms_for_scan_path_index %d\n", n_atoms_for_scan_path_index);

    // int const nodes_offset =
    //     nodes_offset_for_scan_path_for_gen[n_atoms_for_scan_path_index];
    // printf("nodes_offset %d\n", nodes_offset);

    // printf("4\n");
    int const n_atoms_for_scan_path_seg =
        block_type_scan_path_seg_length[block_type][input_conn][first_out_conn]
                                       [gen][scan_path_seg];

    // NOW WE ARE READY!!!
    if (extra_atom_count == 1) {
      // printf(
      //     "Adding in Extra atom for scan path seg %d %d %d %d; line 3127\n",
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg);

      // We must add an extra atom to the nodes tensor for the parent's
      // jump atom
      // UNLESS this is actually the root path, in which case, we
      // have to add node 0.
      int parent_atom_ind = 0;
      if (!is_edge_ft_root) {
        // find the jump atom of the parent block type
        // printf("find the jump atom of the parent block type");
        int const parent_block = ff_edges[pose][ff_edge_on_pose][1];
        int const parent_block_type = pose_stack_block_type[pose][parent_block];
        int const parent_local_jump_atom =
            block_type_jump_atom[parent_block_type];
        // parent_atom_ind = pose * max_n_atoms_per_pose
        //                   + pose_stack_block_coord_offset[pose][parent_block]
        //                   // WRONG!!!
        //                   + parent_local_jump_atom;
        parent_atom_ind =
            atom_kfo_index[pose][parent_block][parent_local_jump_atom];
        // printf(
        //     "parent block %d parent block type %d parent local jump atom %d "
        //     "parent atom ind %d\n",
        //     parent_block,
        //     parent_block_type,
        //     parent_local_jump_atom,
        //     parent_atom_ind);
      }

      // printf(
      //     "Setting extra atom for jump %d %d %d %d e: %d (%d -> %d);"
      //     "nodes[%d] = %d; nodes_bw[%d + %d] = %d;\n",
      //     pose,
      //     block,
      //     gen,
      //     scan_path_seg,
      //     ff_edge_on_pose,
      //     ff_edges[pose][ff_edge_on_pose][1],
      //     ff_edges[pose][ff_edge_on_pose][2],
      //     nodes_offset,
      //     parent_atom_ind,
      //     nodes_offset_bw,
      //     n_atoms_for_scan_path_seg,
      //     parent_atom_ind);

      nodes_fw[nodes_offset] = parent_atom_ind;
      nodes_bw[nodes_offset_bw + n_atoms_for_scan_path_seg] = parent_atom_ind;
    }
    // printf("5\n");

    int const bt_scan_path_seg_start =
        block_type_scan_path_seg_starts[block_type][input_conn][first_out_conn]
                                       [gen][scan_path_seg];
    for (int j = 0; j < n_atoms_for_scan_path_seg; ++j) {
      // printf(
      //     "setting nodes[%d + %d + %d = %d] = %d\n",
      //     nodes_offset,
      //     j,
      //     extra_atom_count,
      //     nodes_offset + j + extra_atom_count,
      //     atom_kfo_index[pose][block]
      //                   [block_type_nodes_for_gens[block_type][input_conn]
      //                                             [first_out_conn][gen]
      //                                             [bt_scan_path_seg_start +
      //                                             j]]);
      // printf(
      //     "nodes_fw[%d + %d + %d] = "
      //     "atom_kfo_index[%d][%d][block_type_nodes_for_gens[%d][%d][%d][%d][%d]"
      //     "+ %d]]; "
      //     "nodes_bw[%d + %d + %d - 1 - %d = %d] = ibid"
      //     "\n",
      //     nodes_offset,
      //     j,
      //     extra_atom_count,
      //     pose,
      //     block,
      //     block_type,
      //     input_conn,
      //     first_out_conn,
      //     gen,
      //     bt_scan_path_seg_start,
      //     j,
      //     nodes_offset_bw,
      //     n_atoms_for_scan_path_seg,
      //     extra_atom_count,
      //     j,
      //     nodes_offset_bw + n_atoms_for_scan_path_seg + extra_atom_count - 1
      //     - j

      // );
      int const j_atom_ind =
          atom_kfo_index[pose][block]
                        [block_type_nodes_for_gens[block_type][input_conn]
                                                  [first_out_conn][gen]
                                                  [bt_scan_path_seg_start + j]];

      nodes_fw[nodes_offset + j + extra_atom_count] = j_atom_ind;
      nodes_bw[nodes_offset_bw + n_atoms_for_scan_path_seg - 1 - j] =
          j_atom_ind;
      // (block_type_nodes_for_gens[block_type][input_conn][first_out_conn]
      //                           [gen][bt_scan_path_start + j]
      //  + pose * max_n_atoms_per_pose
      //  + pose_stack_block_coord_offset[pose][block]);
    }
    if (is_scan_path_seg_root_of_scan_path[sps_index_in_n_atoms_offset]) {
      int const sps_offset = root_scan_path_offset[sps_index_in_n_atoms_offset];
      // printf(
      //     "setting scans[%d] = %d (%d - %d)\n",
      //     sps_offset,
      //     nodes_offset - tsedge0_node_offset,
      //     nodes_offset,
      //     tsedge0_node_offset);
      scans_fw[sps_offset] = nodes_offset - tsedge0_node_offset;
    }
    if (is_scan_path_seg_root_of_scan_path_bw[sps_index_in_n_atoms_offset_bw]) {
      // int const sps_offset =
      // root_scan_path_offset[sps_index_in_n_atoms_offset];
      int const sps_offset_bw =
          root_scan_path_offset_bw[sps_index_in_n_atoms_offset_bw];
      // printf(
      //     "setting scans_bw[%d] = %d (%d - %d)\n",
      //     sps_offset_bw,
      //     nodes_offset_bw - tsedge0_node_offset_bw,
      //     nodes_offset_bw,
      //     tsedge0_node_offset_bw);
      scans_bw[sps_offset_bw] = nodes_offset_bw - tsedge0_node_offset_bw;
    }
  });

  int const n_iter_for_fntfspss = std::max(
      n_poses * max_n_blocks * max_n_gens_per_bt * max_n_scan_path_segs_per_gen,
      n_gens_total + 1);
  DeviceDispatch<D>::template forall<launch_t>(
      n_iter_for_fntfspss, fill_nodes_tensor_from_scan_path_seg_stencils);
  gpuErrPeek;
  gpuErrSync;

  // printf("Step 14: Done\n");

  // for (int i = 0; i < n_nodes_total; ++i) {
  //   printf("nodes[%d] = %d\n", i, nodes[i]);
  // }
  // for (int i = 0; i < n_scan_path_roots_total; ++i) {
  //   printf("scans[%d] = %d\n", i, scans[i]);
  // }
  // for (int i = 0; i < n_gens_total + 1; ++i) {
  //   printf("gens[%d] = %d %d\n", i, gens[i][0], gens[i][1]);
  // }

  // auto copy_scan_ends_to_prev = ([=] TMOL_DEVICE_FUNC (int ind) {
  //   int scan_path_offset = scans[ind][0];
  //   if (ind == 0) {
  //     scans[n_scan_path_roots_total][1] = n_nodes_total;
  //   } else {
  //     scans[ind - 1][1] = scan_path_offset;
  //   }
  // });
  // DeviceDispatch<D>::template forall(n_scan_path_roots_total,
  // copy_scan_ends_to_prev);

  // std::tuple<TPack<Int, 1, D>, TPack<Int, 1, D>>
  return {nodes_fw_t, scans_fw_t, gens_fw_t, nodes_bw_t, scans_bw_t, gens_bw_t};
}

}  // namespace kinematics
}  // namespace tmol

// GARBAGE BELOW??
//   static auto get_parent_atoms(
//     TView<Int, 2, D> ff_block_parent, // Which block is the parent? -1 for
//     root TView<Int, 2, D> ff_conn_to_parent, // What kind of connection:
//     1=lower connect, 2=upper connect, 3=jump TView<Int, 3, D>
//     block_in_and_first_out, // Which connection is the input connection,
//     which the output connection? TView<Int, 2, D>
//     pose_stack_block_coord_offset, TView<Int, 2, D> pose_stack_block_type,

//     TView<Int, 2, D> kfo_block_offset,
//     TView<Int, 2, D> real_bt_ind_for_bt,

//     // For determining which atoms to retrieve from neighboring
//     // residues we have to know how the blocks in the Pose
//     // are connected
//     TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,

//     //////////////////////
//     // Chemical properties
//     // how many atoms for a given block
//     // Dimsize n_block_types
//     TView<Int, 1, D> block_type_n_atoms,
//     // TView<Int, 3, Dev> block_type_atom_downstream_of_conn,

//     // n-bt x max-n-ats x 3 x 3
//     // TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors,

//     // n-bt x max-n-ats x 3 [phi, theta, D]
//     // TView<Real, 3, Dev> block_type_atom_icoors,

//     // TEMP! Handle the case when an atom's coordinate depends on
//     // an un-resolvable atom, e.g., "down" for an N-terminal atom
//     // n-bt x max-n-ats x 3 x 3
//     // TView<UnresolvedAtomID<Int>, 3, Dev> block_type_atom_ancestors_backup,
//     // n-bt x max-n-ats x 3 [phi, theta, D]
//     // TView<Real, 3, Dev> block_type_atom_icoors_backup

//     // the maximum number of atoms in a Pose
//     int const max_n_atoms
//   ) -> TPack<Vec<Real, 3>, 2, Dev>
//   {
//     int const n_poses = ff_block_parent.size(0);
//     TPack<Int, 2, D> parent_atoms = TPack<Int, 2, Dev>::zeros({n_poses,
//     max_n_atoms});

//     auto eval_energies_by_block = ([=] TMOL_DEVICE_FUNC(int ind) {

//         return lj_atom_energy(
//             atom_tile_ind1, atom_tile_ind2, score_dat, cp_separation);
//     });
//   }

// static auto EIGEN_DEVICE_FUNC get_parent(
// ) -> Int {
//   return 0;
// }
