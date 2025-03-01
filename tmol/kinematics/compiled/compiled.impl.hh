#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/TensorStruct.h>
#include <tmol/utility/tensor/TensorUtil.h>
#include <tmol/utility/nvtx.hh>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/uaid_util.hh>
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
  auto get_parent_connections = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / max_n_blocks;
    int const block = i % max_n_blocks;
    int const block_type = pose_stack_block_type[pose][block];
    if (block_type == -1) {
      return;
    }
    int const ff_edge = first_ff_edge_for_block[pose][block];
    int const edge_type = ff_edges[pose][ff_edge][0];
    int const parent_block = pose_stack_ff_parent[pose][block];
    if (parent_block != -1) {
      int const parent_ff_edge = first_ff_edge_for_block[pose][parent_block];
      if (ff_edge == parent_ff_edge) {
        // parent is in the same FF edge
        if (edge_type == ff_polymer_edge) {
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
          pose_stack_block_in_and_first_out[pose][block][0] =
              block_type_n_conn[block_type];
        }
      } else {
        if (edge_type == ff_polymer_edge) {
          // polymer edge
          int conn_to_parent =
              block_type_polymeric_conn_index[block_type]
                                             [(parent_block < block) ? 0 : 1];
          pose_stack_block_in_and_first_out[pose][block][0] = conn_to_parent;

        } else {
          // jump edge
          // assert edge_type == 1
          pose_stack_block_in_and_first_out[pose][block][0] =
              block_type_n_conn[block_type];
        }
      }
    } else {
      // looking at the root block
      // "root connection" index is n_conn + 1
      pose_stack_block_in_and_first_out[pose][block][0] =
          block_type_n_conn[block_type] + 1;
      int const edge_type = ff_edges[pose][ff_edge][0];
      int const end_block = ff_edges[pose][ff_edge][2];
      if (edge_type == ff_polymer_edge) {
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

  // Also handle the first output connection for the end residue of each edge
  auto set_output_conn_for_edge_end = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / max_n_ff_edges_per_pose;
    int const edge = i % max_n_ff_edges_per_pose;
    int const edge_type = ff_edges[pose][edge][0];
    if (edge_type == -1) {
      return;
    }

    int const edge_end_block = ff_edges[pose][edge][2];
    int const block_type = pose_stack_block_type[pose][edge_end_block];
    int const edge_first_child = first_child_of_ff_edge[pose][edge];
    if (edge_first_child != -1) {
      int const first_child_edge_type = ff_edges[pose][edge_first_child][0];
      if (first_child_edge_type == 0) {
        // polymer edge
        int const first_child_end_block = ff_edges[pose][edge_first_child][2];
        pose_stack_block_in_and_first_out[pose][edge_end_block][1] =
            block_type_polymeric_conn_index
                [block_type][(edge_end_block < first_child_end_block) ? 1 : 0];
      } else {
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

  return pose_stack_block_in_and_first_out_t;
}

// P -- number of Poses
// L -- length of the longest Pose
// C -- the maximum number of inter-residue connections
// T -- number of block types
// O -- number of output connection types; i.e. max-n-conn + 2
// A -- maximum number of atoms in a block
template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_kfo_atom_parents(
    TView<Int, 2, D> pose_stack_block_type,                 // P x L
    TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
    TView<Int, 2, D> pose_stack_ff_parent,                  // P x L
    TView<Int, 3, D> pose_stack_block_in_and_first_out,     // P x L x 2
    TView<Int, 3, D> block_type_parents,                    // T x O x A
    TView<Int, 2, D> kfo_2_orig_mapping,                    // K x 3
    TView<Int, 3, D> atom_kfo_index,                        // P x L x A
    TView<Int, 1, D> block_type_jump_atom,                  // T
    TView<Int, 1, D> block_type_n_conn,                     // T
    TView<Int, 2, D> block_type_conn_atom                   // T x C
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

    int const bt_parent_for_atom =
        block_type_parents[block_type][conn_to_parent][atom];
    if (bt_parent_for_atom < 0) {
      // Inter-residue connection
      int const parent_block = pose_stack_ff_parent[pose][block];
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
          int const parent_conn =
              pose_stack_inter_residue_connections[pose][block][conn_to_parent]
                                                  [1];
          int const parent_conn_atom =
              block_type_conn_atom[parent_block_type][parent_conn];
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
    if (parent == 0) {
      // This atom's parent is the root and is connected to it by a jump
      Int one(1);
      accumulate<D, Int>::add(n_jump_children[parent], one);
      is_atom_jump[i] = true;
    } else {
      int const parent_block = kfo_2_orig_mapping[parent][1];
      if (parent_block == block) {
        // Intra-residue connection
        accumulate<D, Int>::add(n_non_jump_children[parent], 1);
      } else {
        // Inter-residue connection, but, is it a jump connetion?
        int const n_conn = block_type_n_conn[block_type];
        int const conn_to_parent =
            pose_stack_block_in_and_first_out[pose][block][0];
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
  DeviceDispatch<D>::template forall<launch_t>(
      n_kfo_atoms, count_children_for_parent);

  auto sum_jump_and_non_jump_children = ([=] TMOL_DEVICE_FUNC(int i) {
    // Now each atom looks at how many jump and non-jump children it has.
    n_children[i] = n_non_jump_children[i] + n_jump_children[i];
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_kfo_atoms, sum_jump_and_non_jump_children);

  // Now get the beginning and end indices for the child-list ranges.
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
    } else {
      int const non_jump_offset =
          accumulate<D, Int>::add(count_n_non_jump_children[parent], 1);
      int const non_jump_start =
          child_list_span[parent] + n_jump_children[parent];
      child_list[non_jump_start + non_jump_offset] = i;
    }
  });
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
          if (a > b) {
            child_list[start + k] = b;
            child_list[start + k + 1] = a;
          }
        }
      }
    }
  });
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
        TPack<Int, 1, D>,
        TPack<bool, 2, D>> {
  LAUNCH_BOX_32;
  int const n_kintree_nodes = parents.size(0);

  auto id_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto frame_x_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto frame_y_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto frame_z_t = TPack<Int, 1, D>::zeros({n_kintree_nodes});
  auto keep_dof_fixed_t = TPack<bool, 2, D>::zeros({n_kintree_nodes, 9});
  auto id = id_t.view;
  auto frame_x = frame_x_t.view;
  auto frame_y = frame_y_t.view;
  auto frame_z = frame_z_t.view;
  auto keep_dof_fixed = keep_dof_fixed_t.view;

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

  // Step 3: mark the DOFs that should always be held fixed:
  // For a "bonded atom", this happens for "theta" when its
  // parent is a jump, and it's the "c1" atom of its parent;
  // thus the atom appears twice in the
  // definition of theta: atom-parent-frame_y[parent] (where
  // frame_y[parent] == atom).
  // It also happens for "phi_p" and "phi_c" if the atom's parent
  // or grand parent is a jump and the atom is the frame_y
  // or frame_z atom.
  // For a "jump atom", this only applies to the root of the kintree
  // (aka the root of the kinforest.)

  int const n_dofs = 9;
  auto mark_fixed_dofs = ([=] TMOL_DEVICE_FUNC(int i) {
    int atom = i / n_dofs;
    int dof = i % n_dofs;
    bool is_jump = is_atom_jump[atom];
    if (is_jump) {
      if (atom == 0) {
        keep_dof_fixed[atom][dof] = true;
      } else if (dof >= 6) {
        // We only minimize the first six dofs for jump atoms
        // in any case.
        keep_dof_fixed[atom][dof] = true;
      }
    } else {
      int parent = parents[atom];
      if (is_atom_jump[parent]) {
        if (frame_y[parent] == atom
            && (dof == bond_dof_theta || dof == bond_dof_phi_p)) {
          // bond_dof_d and bond_dof_phi_c are okay to move here;
          // well, it's possible that phi_c should be held fixed, but
          // that will be determined when we find the grandchild
          // of the jump; phi_c should be disabled here if the jumps'
          // grandchild is jump's the frame_z atom.
          keep_dof_fixed[atom][dof] = true;
        } else if (frame_z[parent] == atom && dof == bond_dof_phi_p) {
          // we might also be concerned about the "phi_c" that would move
          // this atom, except that DOF does not exist: its parent is a
          // jump atom and therefore does not have a phi_c DOF.
          keep_dof_fixed[atom][dof] = true;
        }
      } else {
        int grandparent = parents[parent];
        if (is_atom_jump[grandparent]) {
          if (frame_z[grandparent] == atom) {
            if (dof == bond_dof_phi_c) {
              // NOTE! The dof we are turning off here is the parent's phi_c
              // DOF, not the atom's phi_c DOF. By construction, there is
              // only one value of i that will disable this DOF so no race
              // condition will occur.
              keep_dof_fixed[parent][dof] = true;
            } else if (dof == bond_dof_phi_p) {
              keep_dof_fixed[atom][dof] = true;
            }
          }
        }
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_kintree_nodes * n_dofs, mark_fixed_dofs);

  return {id_t, frame_x_t, frame_y_t, frame_z_t, keep_dof_fixed_t};
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_jump_atom_indices(
    TView<Int, 3, D>
        ff_edges,  // P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
    TView<Int, 2, D> pose_stack_block_type,  // P x L
    TView<Int, 1, D> block_type_jump_atom    // T
    ) -> std::tuple<TPack<Int, 3, D>, TPack<Int, 2, D>> {
  // Get the atom indices for both jumps and root-jumps.
  // Jumps are placed in an n-poses x max-n-edges tensor
  // with the last dimension giving 0) the block index of
  // the downstream end of the jump and 1) the atom index
  // on that block. Root jumps are placed in an
  // n-poses x max-n-blocks tensor where a non-sentinel entry
  // represents the index of the atom on a root-jump block.
  LAUNCH_BOX_32;

  int const n_poses = pose_stack_block_type.size(0);
  int const max_n_blocks = pose_stack_block_type.size(1);
  int const max_n_edges = ff_edges.size(1);  //  too many; could count fewer

  assert(ff_edges.size(0) == n_poses);
  assert(pose_stack_block_type.size(1));

  // Last dimensions:
  // 1: end block
  // 2: atom on end block
  auto pose_stack_atom_for_jump_t =
      TPack<Int, 3, D>::full({n_poses, max_n_edges, 2}, -1);
  auto pose_stack_atom_for_jump = pose_stack_atom_for_jump_t.view;
  auto pose_stack_atom_for_root_jump_t =
      TPack<Int, 2, D>::full({n_poses, max_n_blocks}, -1);
  auto pose_stack_atom_for_root_jump = pose_stack_atom_for_root_jump_t.view;

  auto set_jump_and_root_jump_atom_indices = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / max_n_edges;
    int const edge = i % max_n_edges;
    int const edge_type = ff_edges[pose][edge][0];
    int const jump_ind = ff_edges[pose][edge][3];
    if (edge_type != ff_jump_edge && edge_type != ff_root_jump_edge) {
      return;
    }
    int const end_block = ff_edges[pose][edge][2];
    int const end_block_type = pose_stack_block_type[pose][end_block];
    int const jump_atom = block_type_jump_atom[end_block_type];
    if (edge_type == ff_jump_edge) {
      pose_stack_atom_for_jump[pose][jump_ind][0] = end_block;
      pose_stack_atom_for_jump[pose][jump_ind][1] = jump_atom;
    } else {
      pose_stack_atom_for_root_jump[pose][end_block] = jump_atom;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_edges, set_jump_and_root_jump_atom_indices);

  return {pose_stack_atom_for_jump_t, pose_stack_atom_for_root_jump_t};
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
    // TView<Int, 1, Device::CPU> pose_stack_n_res,  // P
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

  // Step 1:
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
      TPack<Int, 2, Device::CPU>::full({n_poses, max_n_edges_per_ff}, -1);
  auto ff_edge_parent = ff_edge_parent_t.view;

  auto n_ff_edges_t =
      TPack<Int, 1, Device::CPU>::full({n_poses}, max_n_edges_per_ff);
  auto n_ff_edges = n_ff_edges_t.view;

  std::vector<std::vector<std::list<std::tuple<int, int>>>> ff_children(
      n_poses);
  std::vector<std::vector<int>> edge_parent_for_block(n_poses);
  std::vector<std::list<int>> root_jump_blocks(n_poses);
  for (int pose = 0; pose < n_poses; ++pose) {
    ff_children[pose].resize(max_n_blocks);
    edge_parent_for_block[pose].resize(max_n_blocks, -1);
  }
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int edge = 0; edge < max_n_edges_per_ff; ++edge) {
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      if (ff_edge_type == -1) {
        n_ff_edges[pose] =
            edge;  // we are one past the last edge, thus at the number of edges
        continue;
      }
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      int const ff_edge_end = ff_edges_cpu[pose][edge][2];
      if (ff_edge_type == ff_root_jump_edge) {
        assert(ff_edge_start == -1);
        root_jump_blocks[pose].push_back(ff_edge_end);
        edge_parent_for_block[pose][ff_edge_end] = edge;
      } else {
        // The edge that ends at a given block
        edge_parent_for_block[pose][ff_edge_end] = edge;
        ff_children[pose][ff_edge_start].push_back(
            std::make_tuple(ff_edge_end, edge));
      }
    }
    for (int edge = 0; edge < max_n_edges_per_ff; ++edge) {
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      if (ff_edge_type == -1 || ff_edge_type == ff_root_jump_edge) {
        continue;
      }
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      ff_edge_parent[pose][edge] = edge_parent_for_block[pose][ff_edge_start];
    }
  }

  // Now let's perform the depth-first traversals for each pose.
  for (int pose = 0; pose < n_poses; ++pose) {
    int count_dfs_ind = 0;
    std::vector<std::tuple<int, int>> stack;
    for (auto const& root_jump_block : root_jump_blocks[pose]) {
      int const root_jump_edge = edge_parent_for_block[pose][root_jump_block];
      stack.push_back({root_jump_block, root_jump_edge});
    }

    while (!stack.empty()) {
      std::tuple<int, int> const child_edge_tuple = stack.back();
      stack.pop_back();
      int const block = std::get<0>(child_edge_tuple);
      int const edge = std::get<1>(child_edge_tuple);
      dfs_order_of_ff_edges[pose][count_dfs_ind] = edge;
      count_dfs_ind += 1;
      for (auto const& child : ff_children[pose][block]) {
        stack.push_back(child);
      }
    }
  }

  // Step 2:
  // Step N-10:
  // Write down for each residue the first edge in the fold forest that builds
  // it using the partial order of the fold-forest edges. Note that an edge's
  // start residue is not first built by that edge. In the same traversal, let's
  // also calculate the maximum number of generations of any block type of any
  // edge????? OR let's just assume that every edge has the same number of
  // generations for now and TO DO: write a segmented scan on max() to identify
  // the number of generations for each particular residue that is built by an
  // edge.
  // NOTE: root-jump blocks will be left with their "first_ff_edge_for_block"
  // as -1.
  auto first_ff_edge_for_block_cpu_t =
      TPack<Int, 2, Device::CPU>::full({n_poses, max_n_blocks}, -1);
  auto first_ff_edge_for_block_cpu = first_ff_edge_for_block_cpu_t.view;

  auto pose_stack_ff_parent_t =
      TPack<Int, 2, Device::CPU>::full({n_poses, max_n_blocks}, -1);
  auto pose_stack_ff_parent = pose_stack_ff_parent_t.view;

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
      if (ff_edge_type == ff_polymer_edge) {
        int const increment = (ff_edge_start < ff_edge_end) ? 1 : -1;
        int const stop = ff_edge_end + increment;
        int prev_res = ff_edge_start;
        for (int block = ff_edge_start + increment; block != stop;
             block += increment) {
          first_ff_edge_for_block_cpu[pose][block] = edge;
          pose_stack_ff_parent[pose][block] = prev_res;
          prev_res = block;
        }
      } else if (ff_edge_type == ff_jump_edge) {
        // jump edge! The first block is not built by the jump,
        // but the second block is.
        first_ff_edge_for_block_cpu[pose][ff_edge_end] = edge;
        pose_stack_ff_parent[pose][ff_edge_end] = ff_edge_start;
      } else if (ff_edge_type == ff_root_jump_edge) {
        // root jump edge! The first block is not built by the jump,
        // but the second block is.
        // root-jump blocks will be marked as having their parent
        // as -1: "the root"
        assert(ff_edge_start == -1);
        first_ff_edge_for_block_cpu[pose][ff_edge_end] = edge;
        pose_stack_ff_parent[pose][ff_edge_end] = ff_edge_start;
      }
    }
  }

  // Step 3:
  // Step N-9:
  // Find the maximum number of generations of any block type of any edge in the
  // fold forest. TEMP!!!
  auto max_n_gens_for_ff_edge_t = TPack<Int, 2, Device::CPU>::full(
      {n_poses, max_n_edges_per_ff}, max_n_gens_per_bt);
  auto max_n_gens_for_ff_edge = max_n_gens_for_ff_edge_t.view;

  // Step 4:
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

      int const ff_edge_max_n_gens = max_n_gens_for_ff_edge[pose][edge];
      int max_child_gen_depth = -1;
      int second_max_child_gen_depth = -1;
      int first_child = -1;
      for (auto const& child : ff_children[pose][ff_edge_end]) {
        int const child_edge = std::get<1>(child);
        int const child_gen_depth = max_gen_depth_of_ff_edge[pose][child_edge];
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
      max_gen_depth_of_ff_edge[pose][edge] = edge_gen_depth;
    }
  }

  // Step 5:
  // Step N-7:
  // Compute the delay for each edge given the path decomposition of the
  // fold-forest.
  int max_delay = 0;
  for (int pose = 0; pose < n_poses; ++pose) {
    // Now select the first edges to be built from each root-jump block,
    // set the root-jump edge that builds it to have a delay of 0,
    // set its first descendant to have a delay of 0, and set all other
    // edges leaving the root-jump block to have a delay of 1
    for (auto root_jump_block : root_jump_blocks[pose]) {
      int const root_edge = first_ff_edge_for_block_cpu[pose][root_jump_block];
      delay_for_edge[pose][root_edge] = 0;

      int max_root_child_gen_depth = -1;
      int max_root_child_edge = -1;
      for (auto const& child : ff_children[pose][root_jump_block]) {
        int const child_edge = std::get<1>(child);
        int const child_gen_depth = max_gen_depth_of_ff_edge[pose][child_edge];
        if (child_gen_depth > max_root_child_gen_depth) {
          max_root_child_gen_depth = child_gen_depth;
          max_root_child_edge = child_edge;
        }
      }
      if (max_root_child_edge == -1) {
        // Not all root-jump blocks have children
        continue;
      }
      delay_for_edge[pose][max_root_child_edge] = 0;
      for (auto const& child : ff_children[pose][root_jump_block]) {
        int const child_edge = std::get<1>(child);
        if (child_edge == max_root_child_edge) {
          continue;
        }
        delay_for_edge[pose][child_edge] = 1;
        // TO DO: figure out if this is necessary
        // Should not be after explicit root-jump refactor
        if (max_delay < 1) {
          max_delay = 1;
        }
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
    // append all the root-jump edges
    for (auto root_jump_block : root_jump_blocks[pose]) {
      int const root_jump_edge =
          first_ff_edge_for_block_cpu[pose][root_jump_block];
      roots_of_subpaths_by_generation[0].push_back(
          pose * max_n_edges_per_ff + root_jump_edge);
    }
  }
  // Now let's assign a toplogical sort order to each edge:
  // Note that we are using a "global" indexing of the edges, ie
  // one that spans all poses in the PoseStack.
  int topo_sort_ind = 0;
  for (int delay = 0; delay < max_delay + 1; ++delay) {
    for (auto const& root_edge : roots_of_subpaths_by_generation[delay]) {
      int const pose = root_edge / max_n_edges_per_ff;

      int subpath_root_edge = root_edge % max_n_edges_per_ff;
      while (subpath_root_edge != -1) {
        // Write down the next edge in this path,
        // which we will recusively consider the root of
        // another subpath
        topo_sort_index_for_edge
            [pose * max_n_edges_per_ff + subpath_root_edge] = topo_sort_ind;
        topo_sort_ind += 1;
        int const first_child = first_child_of_ff_edge[pose][subpath_root_edge];
        int const subpath_end_block = ff_edges_cpu[pose][subpath_root_edge][2];
        for (auto const& child_edge_pair :
             ff_children[pose][subpath_end_block]) {
          int const next_child_edge = std::get<1>(child_edge_pair);
          if (next_child_edge != first_child) {
            // Write down this edge as the root of another scan path
            // that we will traverse in the next pass
            roots_of_subpaths_by_generation[delay + 1].push_back(
                pose * max_n_edges_per_ff + next_child_edge);
          }
        }
        // Move to the next node in this path
        subpath_root_edge = first_child;
      }
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
  // have to count the number of block-scan paths that come before it. This
  // can be tricky because some block-scan paths continue into other blocks,
  // and we do not know a priori how many block-scan paths there are
  // downstream of such a block-scan path. For each (inter-block) scan path,
  // we have to calculate how many block-scan paths comprise it. Each scan
  // path can be readily identified from the fold forest. Each block type
  // should identify which scan paths are inter-block so it's easy to
  // figure out for each block-scan path extend into other blocks: not all
  // do.

  // Step N-5:

  // Step N-4: count the number of blocks that build each
  // (perhaps-multi-res) scan path.

  // Step N-3: perform a segmented scan on the number of blocks that build
  // each (perhaps-multi-res) scan path.

  // Step N-2: write the number of atoms in each scan path to the
  // appropriate place in the n_atoms_for_scan_path_for_gen tensor.

  // Step N-1: perform a scan on the number of atoms in each scan path to
  // get the nodes tensor offset.

  // Step N: copy the scan path stencils into the nodes tensor, adding the
  // pose-stack- and block- offsets to the atom indices. Note that the
  // upstream jump atom must be added for jump edges that are the roots of
  // paths.

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

  auto n_kin_atoms_offset_for_block_t =
      TPack<Int, 2, D>::zeros({n_poses, max_n_blocks});
  auto n_sps_for_ffedge_for_gen_by_topo_sort_t =
      TPack<Int, 2, D>::zeros({n_gens_total, n_poses * max_n_edges_per_ff});
  auto n_sps_for_ffedge_for_gen_segment_starts_t =
      TPack<Int, 1, D>::zeros({n_gens_total});
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

        int const ff_edge_end = ff_edges[pose][edge][2];
        int const end_bt = pose_stack_block_type[pose][ff_edge_end];
        int const end_bt_n_conn = block_type_n_conn[end_bt];
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
          is_edge_end_block_scan_path_seg_root_of_bw_scan_path[pose]
                                                              [ff_edge_end]
                                                              [j_gen][j_sps] =
                                                                  true;
        }
      });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_edges_per_ff,
      mark_ff_edge_end_block_output_conns_as_potential_bw_sp_roots);

  auto is_ff_edge_root_of_scan_path_t =
      TPack<bool, 2, D>::zeros({n_poses, max_n_edges_per_ff});
  auto is_ff_edge_root_of_fold_tree_t =
      TPack<bool, 2, D>::zeros({n_poses, max_n_edges_per_ff});
  auto is_ff_edge_root_of_scan_path_bw_t =
      TPack<bool, 2, D>::ones({n_poses, max_n_edges_per_ff});

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
    if (ff_edge_type == ff_root_jump_edge) {
      // we are looking at the root of the fold tree
      is_ff_edge_root_of_fold_tree[pose][edge] = true;
      is_ff_edge_root_of_scan_path[pose][edge] = true;
    } else {
      int const ff_edge_start = ff_edges[pose][edge][1];
      int const ff_edge_end = ff_edges[pose][edge][2];
      int const first_edge_for_start =
          first_ff_edge_for_block[pose][ff_edge_start];
      int const ff_edge_delay = delay_for_edge[pose][edge];
      int const first_edge_delay = delay_for_edge[pose][first_edge_for_start];

      if (ff_edge_delay != first_edge_delay) {
        // this edge is not the first child of the parent edge
        // which means it must root its own scan path
        is_ff_edge_root_of_scan_path[pose][edge] = true;
      } else {
        // the parent edge continues on into this edge
        // so mark "first_edge_for_start" as not a root of a backwards
        // scan path
        is_ff_edge_root_of_scan_path_bw[pose][first_edge_for_start] = false;
      }

      // Find the SPS on the end block of first_edge_for_start / start
      // block of "edge" that connects it to the next residue on the edge
      // and mark it as NOT being a root of a backwards scan path.
      int const start_bt = pose_stack_block_type[pose][ff_edge_start];
      if (ff_edge_type == ff_jump_edge) {
        // jump edge: noop
      } else {
        // polymer edge: are we going from N->C or C->N?
        int const conn_ind = (ff_edge_start < ff_edge_end) ? 1 : 0;
        int const in_conn =
            pose_stack_block_in_and_first_out[pose][ff_edge_start][0];
        int const out_conn =
            pose_stack_block_in_and_first_out[pose][ff_edge_start][1];
        int const gen =
            block_type_kts_conn_info[start_bt][in_conn][out_conn][conn_ind][0];
        int const sps =
            block_type_kts_conn_info[start_bt][in_conn][out_conn][conn_ind][1];
        if (gen != -1) {
          is_edge_end_block_scan_path_seg_root_of_bw_scan_path[pose]
                                                              [ff_edge_start]
                                                              [gen][sps] =
                                                                  false;
        }
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_edges_per_ff, mark_ff_edge_as_root_of_scan_path);

  // Step 7
  // Mark the scan-path segments that root each (jump & non-jump) fold-forest
  // edge. This will store the per-pose indexing of the fold-forest edge rather
  // than the global indexing, but they can be interconverted easily:
  // pose_ff_edge_index = global_edge_index % max_n_edges_per_ff
  // global_edge_index = pose * max_n_edges_per_ff + pose_ff_edge_index

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
    if (ff_edge_type == ff_jump_edge) {
      // Jump edge
      // A jump edge uses only one atom of the start block
      // and we will append that atom to the nodes list for
      // the first scan path of the end block. We need not
      // look up the scan path on the end block that builds
      // this edge because it will always be the first, but
      // we do need to know whether we are looking at the root
      // of the fold tree.
      // NEW LOGIC: a jump edge is never the root of the fold
      // tree

      jump_ff_edge_rooted_at_scan_path_seg[pose][ff_edge_end][0][0] = edge;
      jump_ff_edge_rooted_at_scan_path_seg_bw[pose][ff_edge_end][0][0] = edge;
    } else if (ff_edge_type == ff_root_jump_edge) {
      jump_ff_edge_rooted_at_scan_path_seg[pose][ff_edge_end][0][0] = edge;
      jump_ff_edge_rooted_at_scan_path_seg_bw[pose][ff_edge_end][0][0] = edge;
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
      // TO DO: remove "is_root" condition; guaranteed to be false
      if (exiting_scan_path_seg_gen != 0) {
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
        // it will not, but in any event, this edge will not. (will not what?)
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

  // Step 8
  // Step N-4:
  // Count the number of scan-path segs that build each ff-edge for
  // each generation with edges ordered by their topological-sort index
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
        // if, e.g., the edge is a jump/root-jump and the start
        // block would have already been built by another edge.
        int const ff_edge_start = ff_edges[pose][edge][1];
        int const ff_edge_end = ff_edges[pose][edge][2];
        int const n_blocks =
            (edge_type == ff_polymer_edge ? (
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

  // Step 10
  // Step N-3:
  // Now, run scan on n_blocks_that_build_edge_for_gen to get
  // block_offset_for_tsedge_for_gen
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

  auto is_scan_path_seg_root_of_scan_path_t = TPack<Int, 1, D>::full(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen}, 0);
  auto is_scan_path_seg_root_of_scan_path_bw_t = TPack<Int, 1, D>::full(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen}, 0);
  auto is_scan_path_seg_root_of_scan_path =
      is_scan_path_seg_root_of_scan_path_t.view;
  auto is_scan_path_seg_root_of_scan_path_bw =
      is_scan_path_seg_root_of_scan_path_bw_t.view;

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
        // 5, and for the edge 25->10, block 24 is at index 1.
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
  auto n_atoms_for_scan_path_seg_for_gen_t = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto n_atoms_for_scan_path_seg_for_gen_bw_t = TPack<Int, 1, D>::zeros(
      {n_blocks_building_edges_total * max_n_scan_path_segs_per_gen});
  auto n_scan_paths_for_gen_t = TPack<Int, 1, D>::zeros({n_gens_total + 1});
  auto n_scan_paths_for_gen_bw_t = TPack<Int, 1, D>::zeros({n_gens_total + 1});

  auto temp_n_nodes_for_gen_t = TPack<Int, 1, D>::zeros({n_gens_total + 1});
  auto temp_n_scan_paths_for_gen_t =
      TPack<Int, 1, D>::zeros({n_gens_total + 1});
  auto temp_n_nodes_for_gen_bw_t = TPack<Int, 1, D>::zeros({n_gens_total + 1});
  auto temp_n_scan_paths_for_gen_bw_t =
      TPack<Int, 1, D>::zeros({n_gens_total + 1});

  auto n_atoms_for_scan_path_seg_for_gen =
      n_atoms_for_scan_path_seg_for_gen_t.view;
  auto n_atoms_for_scan_path_seg_for_gen_bw =
      n_atoms_for_scan_path_seg_for_gen_bw_t.view;
  auto n_scan_paths_for_gen = n_scan_paths_for_gen_t.view;
  auto n_scan_paths_for_gen_bw = n_scan_paths_for_gen_bw_t.view;

  auto temp_n_nodes_for_gen = temp_n_nodes_for_gen_t.view;
  auto temp_n_scan_paths_for_gen = temp_n_scan_paths_for_gen_t.view;
  auto temp_n_nodes_for_gen_bw = temp_n_nodes_for_gen_bw_t.view;
  auto temp_n_scan_paths_for_gen_bw = temp_n_scan_paths_for_gen_bw_t.view;

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

    int const block_type = pose_stack_block_type[pose][block];
    if (block_type == -1) {
      return;
    }

    // During the (gen 0, scan-path-seg 0) iteration, record the number of atoms
    // for this block -- IS THIS REALLY NEEDED??? WE ALREADY HAVE
    // atom_ind_2_kfo_index

    int const input_conn = pose_stack_block_in_and_first_out[pose][block][0];
    int const first_out_conn =
        pose_stack_block_in_and_first_out[pose][block][1];
    if (scan_path_seg >= block_type_n_scan_path_segs[block_type][input_conn]
                                                    [first_out_conn][gen]) {
      return;
    }

    // Note again: "scan path" -- a contiguous, possibly-multi-block stretch of
    // atoms to be updated together vs "scan path segment" the portion of a scan
    // path belonging to a single block. Some scan path segments are scan paths;
    // ie. they start and stop within the same block.
    bool is_root_of_scan_path = false;
    bool is_root_of_scan_path_bw = false;

    if (gen != 0 || scan_path_seg != 0) {
      is_root_of_scan_path = true;
      is_root_of_scan_path_bw = true;
    }

    int ff_edge_on_pose = first_ff_edge_for_block[pose][block];
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
      ff_edge_on_pose = nj_ff_edge_rooted_at_scan_path_seg;
      ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;
      if (is_ff_edge_root_of_scan_path[pose][ff_edge_on_pose]) {
        is_root_of_scan_path = true;
      }
      if (!is_edge_end_block_scan_path_seg_root_of_bw_scan_path
              [pose][block][gen][scan_path_seg]) {
        is_root_of_scan_path_bw = false;
      }
      if (is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose]) {
        // The scan path leaving the root of the fold forest (atom 0)
        // requires an extra atom that will not be listed in the
        // block-type's-scan path, so we add it here.
        // THIS SHOULD NOW BE IMPOSSIBLE
        is_root_path = true;
        extra_atom_count = 1;
      }
    }
    if (nj_ff_edge_rooted_at_scan_path_seg_bw != -1) {
      assert(ff_edge_on_pose == nj_ff_edge_rooted_at_scan_path_seg_bw);
      assert(
          ff_edge_global_index == ff_edge_on_pose + pose * max_n_edges_per_ff);
      if (is_ff_edge_root_of_scan_path_bw[pose][ff_edge_on_pose]) {
        is_root_of_scan_path_bw = true;
      }
    }

    int const ff_edge_type = ff_edges[pose][ff_edge_on_pose][0];
    if (ff_edge_type == ff_jump_edge || ff_edge_type == ff_root_jump_edge) {
      int const j_ff_edge_rooted_at_scan_path_seg =
          jump_ff_edge_rooted_at_scan_path_seg[pose][block][gen][scan_path_seg];
      if (j_ff_edge_rooted_at_scan_path_seg != -1) {
        ff_edge_on_pose = j_ff_edge_rooted_at_scan_path_seg;
        ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;

        is_root_path = is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose];
        assert(is_root_path == (ff_edge_type == ff_root_jump_edge));
        if (is_ff_edge_root_of_scan_path[pose][ff_edge_on_pose]) {
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
          is_root_of_scan_path = true;
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
          is_root_of_scan_path_bw = true;
        }
      }
    }

    int const ff_edge_gen = gen + ff_edge_delay;
    int const ff_edge_gen_bw = (n_gens_total - 1) - ff_edge_gen;
    int block_position_on_ff_edge = 0;
    int block_position_on_ff_edge_bw = 0;
    if (ff_edge_type == ff_jump_edge || ff_edge_type == ff_root_jump_edge) {
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

    int const edge_toposort_index =
        topo_sort_index_for_edge[ff_edge_global_index];
    int const edge_toposort_index_bw =
        n_poses * max_n_edges_per_ff - 1 - edge_toposort_index;

    int boftsfg = block_offset_for_tsedge_for_gen
        [ff_edge_gen * n_poses * max_n_edges_per_ff + edge_toposort_index];
    int boftsfg_bw = block_offset_for_tsedge_for_gen_bw
        [ff_edge_gen_bw * n_poses * max_n_edges_per_ff
         + edge_toposort_index_bw];

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

    n_atoms_for_scan_path_seg_for_gen[sps_index_in_n_atoms_offset] =
        n_atoms_for_scan_path_seg + extra_atom_count;  // ...TADA!

    n_atoms_for_scan_path_seg_for_gen_bw[sps_index_in_n_atoms_offset_bw] =
        n_atoms_for_scan_path_seg + extra_atom_count;

    if (is_root_of_scan_path) {
      is_scan_path_seg_root_of_scan_path[sps_index_in_n_atoms_offset] = 1;
      accumulate<D, Int>::add(n_scan_paths_for_gen[ff_edge_gen], 1);
    }
    if (is_root_of_scan_path_bw) {
      is_scan_path_seg_root_of_scan_path_bw[sps_index_in_n_atoms_offset_bw] = 1;
      accumulate<D, Int>::add(n_scan_paths_for_gen_bw[ff_edge_gen_bw], 1);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks * max_n_gens_per_bt * max_n_scan_path_segs_per_gen,
      collect_n_atoms_for_scan_path_segs);

  // Step 12
  // And with the number of atoms for each scan path segment, we can now
  // calculate their offsets in the nodes tensor using scan
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

  auto temp_nodes_offset_for_gen_t =
      TPack<Int, 1, D>::zeros({n_gens_total + 1});
  auto temp_nodes_offset_for_gen_bw_t =
      TPack<Int, 1, D>::zeros({n_gens_total + 1});

  auto nodes_offset_for_scan_path_seg_for_gen =
      nodes_offset_for_scan_path_seg_for_gen_tp.view;
  auto nodes_offset_for_scan_path_seg_for_gen_bw =
      nodes_offset_for_scan_path_seg_for_gen_bw_tp.view;
  auto root_scan_path_offset = root_scan_path_offset_tp.view;
  auto root_scan_path_offset_bw = root_scan_path_offset_bw_tp.view;
  auto n_scan_path_offsets_for_gen = n_scan_path_offsets_for_gen_t.view;
  auto n_scan_path_offsets_for_gen_bw = n_scan_path_offsets_for_gen_bw_t.view;

  auto temp_nodes_offset_for_gen = temp_nodes_offset_for_gen_t.view;
  auto temp_nodes_offset_for_gen_bw = temp_nodes_offset_for_gen_bw_t.view;

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

  DeviceDispatch<D>::template scan<mgpu::scan_type_exc>(
      temp_n_nodes_for_gen.data(),
      temp_nodes_offset_for_gen.data(),
      n_gens_total + 1,
      mgpu::plus_t<Int>());
  DeviceDispatch<D>::template scan<mgpu::scan_type_exc>(
      temp_n_nodes_for_gen_bw.data(),
      temp_nodes_offset_for_gen_bw.data(),
      n_gens_total + 1,
      mgpu::plus_t<Int>());

  // Step 13
  // Step N:
  // And we can now, finally, copy the scan-path-segment stencils into
  // the nodes tensor
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
    }

    if (pose >= n_poses) {
      // it is possible, though unlikely, that the max(n_segments, n_gens_total
      // + 1) where n_segments = n_poses * max_n_blocks * max_n_gens_per_bt *
      // max_n_scan_path_segs_per_gen is n_gens_total+1, and so we must check
      // that this thread index is in bounds before proceeding.
      return;
    }

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
      return;
    }

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
      ff_edge_on_pose = nj_ff_edge_rooted_at_scan_path_seg;
      ff_edge_global_index = ff_edge_on_pose + pose * max_n_edges_per_ff;
      is_edge_ft_root = is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose];
      if (is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose]) {
        // The path leaving the root of the fold forest (atom 0)
        // requires an extra atom that will not be listed in the
        // block-type's-scan path, so we add it here.
        // THIS SHOULD NOW BE IMPOSSIBLE
        extra_atom_count = 1;
      }
    }
    int const ff_edge_type = ff_edges[pose][ff_edge_on_pose][0];
    if (ff_edge_type == ff_jump_edge || ff_edge_type == ff_root_jump_edge) {
      int const j_ff_edge_rooted_at_scan_path_seg =
          jump_ff_edge_rooted_at_scan_path_seg[pose][block][gen][scan_path_seg];
      if (j_ff_edge_rooted_at_scan_path_seg != -1) {
        is_edge_ft_root = is_ff_edge_root_of_fold_tree[pose][ff_edge_on_pose];
        if (is_ff_edge_root_of_scan_path[pose][ff_edge_on_pose]) {
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
    }
    int const ff_edge_gen = gen + ff_edge_delay;
    int const ff_edge_gen_bw = (n_gens_total - 1) - ff_edge_gen;
    int block_position_on_ff_edge = 0;
    int block_position_on_ff_edge_bw = 0;
    if (ff_edge_type == ff_jump_edge || ff_edge_type == ff_root_jump_edge) {
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

    int edge_toposort_index = topo_sort_index_for_edge[ff_edge_global_index];
    int const edge_toposort_index_bw =
        n_poses * max_n_edges_per_ff - 1 - edge_toposort_index;
    int boftsfg = block_offset_for_tsedge_for_gen
        [ff_edge_gen * n_poses * max_n_edges_per_ff + edge_toposort_index];
    int boftsfg_bw = block_offset_for_tsedge_for_gen_bw
        [ff_edge_gen_bw * n_poses * max_n_edges_per_ff
         + edge_toposort_index_bw];

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

    int sps_index_in_n_atoms_offset =
        (block_position_on_ff_edge + boftsfg) * max_n_scan_path_segs_per_gen
        + scan_path_seg;
    int sps_index_in_n_atoms_offset_bw =
        (block_position_on_ff_edge_bw + boftsfg_bw)
            * max_n_scan_path_segs_per_gen
        + scan_path_seg;

    int const nodes_offset =
        nodes_offset_for_scan_path_seg_for_gen[sps_index_in_n_atoms_offset];
    int const nodes_offset_bw = nodes_offset_for_scan_path_seg_for_gen_bw
        [sps_index_in_n_atoms_offset_bw];

    int const n_atoms_for_scan_path_seg =
        block_type_scan_path_seg_length[block_type][input_conn][first_out_conn]
                                       [gen][scan_path_seg];

    // NOW WE ARE READY!!!
    if (extra_atom_count == 1) {
      // We must add an extra atom to the nodes tensor for the parent's
      // jump atom
      // UNLESS this is actually the root path, in which case, we
      // have to add node 0.
      int parent_atom_ind = 0;
      if (!is_edge_ft_root) {
        // find the jump atom of the parent block type
        int const parent_block = ff_edges[pose][ff_edge_on_pose][1];
        int const parent_block_type = pose_stack_block_type[pose][parent_block];
        int const parent_local_jump_atom =
            block_type_jump_atom[parent_block_type];
        parent_atom_ind =
            atom_kfo_index[pose][parent_block][parent_local_jump_atom];
      }

      nodes_fw[nodes_offset] = parent_atom_ind;
      nodes_bw[nodes_offset_bw + n_atoms_for_scan_path_seg] = parent_atom_ind;
    }

    int const bt_scan_path_seg_start =
        block_type_scan_path_seg_starts[block_type][input_conn][first_out_conn]
                                       [gen][scan_path_seg];
    for (int j = 0; j < n_atoms_for_scan_path_seg; ++j) {
      int const j_atom_ind =
          atom_kfo_index[pose][block]
                        [block_type_nodes_for_gens[block_type][input_conn]
                                                  [first_out_conn][gen]
                                                  [bt_scan_path_seg_start + j]];

      nodes_fw[nodes_offset + j + extra_atom_count] = j_atom_ind;
      nodes_bw[nodes_offset_bw + n_atoms_for_scan_path_seg - 1 - j] =
          j_atom_ind;
    }
    if (is_scan_path_seg_root_of_scan_path[sps_index_in_n_atoms_offset]) {
      int const sps_offset = root_scan_path_offset[sps_index_in_n_atoms_offset];
      scans_fw[sps_offset] = nodes_offset - tsedge0_node_offset;
    }
    if (is_scan_path_seg_root_of_scan_path_bw[sps_index_in_n_atoms_offset_bw]) {
      int const sps_offset_bw =
          root_scan_path_offset_bw[sps_index_in_n_atoms_offset_bw];
      scans_bw[sps_offset_bw] = nodes_offset_bw - tsedge0_node_offset_bw;
    }
  });

  int const n_iter_for_fntfspss = std::max(
      n_poses * max_n_blocks * max_n_gens_per_bt * max_n_scan_path_segs_per_gen,
      n_gens_total + 1);
  DeviceDispatch<D>::template forall<launch_t>(
      n_iter_for_fntfspss, fill_nodes_tensor_from_scan_path_seg_stencils);

  return {nodes_fw_t, scans_fw_t, gens_fw_t, nodes_bw_t, scans_bw_t, gens_bw_t};
}

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::create_minimizer_map(
    TView<Int, 1, D> kinforest_id,  // K
    int64_t max_n_atoms_per_pose_in,
    TView<Int, 2, D> pose_stack_block_coord_offset,
    TView<Int, 2, D> pose_stack_block_type,
    TView<Vec<Int, 2>, 3, D> pose_stack_inter_block_connections,
    TView<Int, 3, D> pose_stack_block_in_and_first_out,  // P x L x 2
    TView<Int, 3, D> pose_stack_atom_for_jump,           // P x E x 2
    TView<Int, 2, D> pose_stack_atom_for_root_jump,      // P x L
    TView<bool, 2, D> keep_dof_fixed,                    // K x 9
    TView<Int, 1, D> bt_n_named_torsions,
    TView<UnresolvedAtomID<Int>, 3, D> bt_uaid_for_torsion,
    TView<bool, 2, D> bt_named_torsion_is_mc,
    TView<Int, 2, D> bt_which_mcsc_torsion_for_named_torsion,
    TView<Int, 3, D> bt_atom_downstream_of_conn,
    bool move_all_jumps,
    bool move_all_root_jumps,
    bool move_all_mcs,
    bool move_all_scs,
    bool move_all_named_torsions,
    bool non_ideal,
    TView<bool, 2, D> move_jumps,
    TView<bool, 2, D> move_jumps_mask,
    TView<bool, 2, D> move_root_jumps,
    TView<bool, 2, D> move_root_jumps_mask,
    TView<bool, 2, D> move_mcs,
    TView<bool, 2, D> move_mcs_mask,
    TView<bool, 2, D> move_scs,
    TView<bool, 2, D> move_scs_mask,
    TView<bool, 2, D> move_named_torsions,
    TView<bool, 2, D> move_named_torsions_mask,
    TView<bool, 3, D> move_jump_dof,
    TView<bool, 3, D> move_jump_dof_mask,
    TView<bool, 3, D> move_root_jump_dof,
    TView<bool, 3, D> move_root_jump_dof_mask,
    TView<bool, 3, D> move_mc,
    TView<bool, 3, D> move_mc_mask,
    TView<bool, 3, D> move_sc,
    TView<bool, 3, D> move_sc_mask,
    TView<bool, 3, D> move_named_torsion,
    TView<bool, 3, D> move_named_torsion_mask,
    TView<bool, 4, D> move_atom_dof,
    TView<bool, 4, D> move_atom_dof_mask) -> TPack<bool, 2, D> {
  // "Optimal" launch-box size untested; going w/ nt=32, vt=1
  using namespace score::common;
  LAUNCH_BOX_32;

  int const n_poses = pose_stack_block_type.size(0);
  int const n_blocks = pose_stack_block_type.size(1);
  int const n_kinforest_atoms = kinforest_id.size(0);
  int const max_n_jumps_per_pose = pose_stack_atom_for_jump.size(1);
  int const max_n_input_conn_types = bt_uaid_for_torsion.size(1);
  int const max_n_torsions = bt_uaid_for_torsion.size(2);
  int const max_n_atoms_per_block = move_atom_dof.size(2);
  int const max_n_atoms_per_pose = max_n_atoms_per_pose_in;  // HACK!

  auto pose_atom_ordered_minimizer_map_t =
      TPack<bool, 2, D>::zeros({n_poses * max_n_atoms_per_pose, 9});
  auto pose_atom_ordered_minimizer_map = pose_atom_ordered_minimizer_map_t.view;
  auto minimizer_map_t = TPack<bool, 2, D>::zeros({n_kinforest_atoms, 9});
  auto minimizer_map = minimizer_map_t.view;

  // Step 1:
  // resolve where all the torsions live on all the blocks.
  // Dunbrack does this.

  // Step 2: torsions
  // For each torsion, set move status based on whether it is
  // mainchain or sidechain: look at all three levels of status
  // and pick the one that is most restrictive

  // Step 3: jumps
  // For each jump, set its move status based on all three
  // levels of status and pick the one that's most restrictive

  // Step 4: root jumps
  // For each residue, determine if it's a root jump, and if so,
  // activate its jump-atom DOFs according to the
  // "move_all_root_jumps", "moove_root_jumps," and "move_root_jump_dof"
  // instructions.

  // Step 5: atoms
  // For each DOF on each atom, look at whether the "move_atom_dof"
  // parameter has been set, and if so, override whatever else has
  // been set for that DOF.

  // Step 6: reindex
  // Finally, map the DOFs from their pose-stack order to their kinforest
  // order in the minimizer_map tensor; while doing this, override any
  // setting for if keep_dof_fixed is set to true.

  // Step 1:
  auto atom_for_torsion_t =
      TPack<Int, 4, D>::full({n_poses, n_blocks, max_n_torsions, 2}, -1);
  auto atom_for_torsion = atom_for_torsion_t.view;

  auto resolve_torsion_location = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / (n_blocks * max_n_torsions);
    i = i - pose * n_blocks * max_n_torsions;
    int const block = i / max_n_torsions;
    int const torsion = i % max_n_torsions;

    int const block_type = pose_stack_block_type[pose][block];
    if (block_type < 0) {
      return;
    }
    int const n_torsions = bt_n_named_torsions[block_type];
    if (torsion >= n_torsions) {
      return;
    }

    int const in_conn = pose_stack_block_in_and_first_out[pose][block][0];
    UnresolvedAtomID<Int> uaid =
        bt_uaid_for_torsion[block_type][in_conn][torsion];

    // Now resolve the atom index for this torsion; given
    // by the pose-stack-index: this may be an atom on this residue
    // or on anotehr residue
    auto resolved_ind = score::common::resolve_local_atom_ind_from_uaid(
        uaid,
        block,
        pose,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        bt_atom_downstream_of_conn);
    int const tor_atom_block = std::get<0>(resolved_ind);
    int const tor_atom = std::get<1>(resolved_ind);
    atom_for_torsion[pose][block][torsion][0] = tor_atom_block;
    atom_for_torsion[pose][block][torsion][1] = tor_atom;
  });

  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * n_blocks * max_n_torsions, resolve_torsion_location);

  // Step 2:
  auto set_torsion_freedom = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / (n_blocks * max_n_torsions);
    i = i - pose * n_blocks * max_n_torsions;
    int const block = i / max_n_torsions;
    int const torsion = i % max_n_torsions;

    int const block_type = pose_stack_block_type[pose][block];
    if (block_type < 0) {
      return;
    }
    int const n_torsions = bt_n_named_torsions[block_type];
    if (torsion >= n_torsions) {
      return;
    }

    int const tor_atom_block = atom_for_torsion[pose][block][torsion][0];
    int const tor_atom = atom_for_torsion[pose][block][torsion][1];
    if (tor_atom_block < 0) {
      return;
    }

    int const tor_atom_global_index =
        pose * max_n_atoms_per_pose
        + pose_stack_block_coord_offset[pose][tor_atom_block] + tor_atom;
    int const which_mcsc_torsion =
        bt_which_mcsc_torsion_for_named_torsion[block_type][torsion];

    auto heirarchy_of_specifications = ([&] TMOL_DEVICE_FUNC(
                                            auto const& move_mcsc_tor_mask,
                                            auto const& move_mcsc_tor,
                                            auto const& move_mcs_scs_mask,
                                            auto const& move_mcs_scs,
                                            auto move_all_mcs_scs) {
      bool setting;
      if (move_named_torsion_mask[pose][block][torsion]) {
        // First: Look whether there are instructions specifically for this
        // named torsion
        setting = move_named_torsion[pose][block][torsion];
      } else if (move_mcsc_tor_mask[pose][block][which_mcsc_torsion]) {
        // Next: did we have "move mc/sc" instructions for this torsion among
        // the set of mc/sc torsions
        setting = move_mcsc_tor[pose][block][which_mcsc_torsion];
      } else if (move_mcs_scs_mask[pose][block]) {
        // Next: look at the "move mcs/scs" directives for this block
        setting = move_mcs_scs[pose][block];
      } else {
        // Otherwise, fall back on the "global" settings
        setting = move_all_mcs_scs || move_all_named_torsions;
      }

      pose_atom_ordered_minimizer_map[tor_atom_global_index][bond_dof_phi_c] =
          setting;
      if (non_ideal) {
        for (int i = 0; i < 3; ++i) {
          // additionally enable child-phi, theta, and d DOFs for this atom
          pose_atom_ordered_minimizer_map[tor_atom_global_index][i] = setting;
        }
      }
    });

    if (bt_named_torsion_is_mc[block_type][torsion]) {
      heirarchy_of_specifications(
          move_mc_mask, move_mc, move_mcs_mask, move_mcs, move_all_mcs);
    } else {
      heirarchy_of_specifications(
          move_sc_mask, move_sc, move_scs_mask, move_scs, move_all_scs);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * n_blocks * max_n_torsions, set_torsion_freedom);

  //  Step 3:
  auto set_jump_freedom = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / max_n_jumps_per_pose;
    int const jump = i % max_n_jumps_per_pose;
    int const block = pose_stack_atom_for_jump[pose][jump][0];
    int const atom = pose_stack_atom_for_jump[pose][jump][1];
    if (block == -1) {
      return;
    }

    int const jump_atom_global_index =
        pose * max_n_atoms_per_pose + pose_stack_block_coord_offset[pose][block]
        + atom;

    // Now we look at the specification heirarchy for this jump's 6 DOFs
    for (int jump_dof = 0; jump_dof < 6; ++jump_dof) {
      if (move_jump_dof_mask[pose][jump][jump_dof]) {
        pose_atom_ordered_minimizer_map[jump_atom_global_index][jump_dof] =
            move_jump_dof[pose][jump][jump_dof];
      } else if (move_jumps_mask[pose][jump]) {
        pose_atom_ordered_minimizer_map[jump_atom_global_index][jump_dof] =
            move_jumps[pose][jump];
      } else {
        pose_atom_ordered_minimizer_map[jump_atom_global_index][jump_dof] =
            move_all_jumps;
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_jumps_per_pose, set_jump_freedom);

  //  Step 4:
  auto set_root_jump_freedom = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / n_blocks;
    int const block = i % n_blocks;
    int const atom = pose_stack_atom_for_root_jump[pose][block];
    if (atom == -1) {
      return;
    }

    int const jump_atom_global_index =
        pose * max_n_atoms_per_pose + pose_stack_block_coord_offset[pose][block]
        + atom;

    // Now we look at the specification heirarchy for this root-jump's 6 DOFs
    for (int jump_dof = 0; jump_dof < 6; ++jump_dof) {
      if (move_root_jump_dof_mask[pose][block][jump_dof]) {
        pose_atom_ordered_minimizer_map[jump_atom_global_index][jump_dof] =
            move_root_jump_dof[pose][block][jump_dof];
      } else if (move_root_jumps_mask[pose][block]) {
        pose_atom_ordered_minimizer_map[jump_atom_global_index][jump_dof] =
            move_root_jumps[pose][block];
      } else {
        pose_atom_ordered_minimizer_map[jump_atom_global_index][jump_dof] =
            move_all_root_jumps;
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * n_blocks, set_root_jump_freedom);

  // Step 5:
  auto set_atom_freedom = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose = i / (n_blocks * max_n_atoms_per_block);
    i = i - pose * n_blocks * max_n_atoms_per_block;
    int const block = i / max_n_atoms_per_block;
    int const atom = i % max_n_atoms_per_block;

    for (int dof = 0; dof < 3; ++dof) {
      if (move_atom_dof_mask[pose][block][atom][dof]) {
        pose_atom_ordered_minimizer_map
            [pose * max_n_atoms_per_pose
             + pose_stack_block_coord_offset[pose][block] + atom][dof] =
                move_atom_dof[pose][block][atom][dof];
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * n_blocks * max_n_atoms_per_block, set_atom_freedom);

  // Step 6:
  auto reindex_minimizer_map = ([=] TMOL_DEVICE_FUNC(int i) {
    int const pose_atom = kinforest_id[i];
    if (i > 0) {
      int const pose = pose_atom / max_n_atoms_per_pose;
      int const atom = pose_atom % max_n_atoms_per_pose;
      for (int dof = 0; dof < 9; ++dof) {
        if (keep_dof_fixed[i][dof]) {
          minimizer_map[i][dof] = 0;
        } else {
          minimizer_map[i][dof] =
              pose_atom_ordered_minimizer_map[pose_atom][dof];
        }
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_kinforest_atoms, reindex_minimizer_map);

  return minimizer_map_t;
}

}  // namespace kinematics
}  // namespace tmol
