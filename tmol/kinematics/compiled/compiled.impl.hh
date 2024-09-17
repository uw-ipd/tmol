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

// namespace compiled {

// template <
//     template <tmol::Device>
//     class DeviceDispatch,
//     tmol::Device D,
//     typename Real,
//     typename Int>
// auto KinDerivDispatch<DeviceDispatch, D, Real, Int>::f(
//     TView<Int, 1, D> parents,
//     TView<Int, 1, D> frame_x,
//     TView<Int, 1, D> frame_y,
//     TView<Int, 1, D> frame_z,
//     TView<Int, 1, D> roots,
//     TView<Int, 1, D> jumps
// )
// {
//     int const n_kintree_nodes = parents.size(0);
//     int const n_roots = roots.size(0);
//     int const n_jumps = jumps.size(0);

//     assert(frame_x.size(0) == n_kintree_nodes);
//     assert(frame_y.size(0) == n_kintree_nodes);
//     assert(frame_z.size(0) == n_kintree_nodes);

//     // Step 1: construct child-list and child-list spans
//     auto child_list_t = TPack<Int, 1, D>::zeros({parents.size()});
//     auto child_list_span_t = TPack<Int, 1, D>::zeros({parents.size() + 1});
//     auto n_children_t = TPack<Int, 1, D>::zeros({parents.size() + 1});
//     auto count_children_added_t = TPack<Int, 1, D>::zeros({parents.size()});

//     auto child_list = child_list_t.view;
//     auto child_list_span = child_list_span_t.view;
//     auto n_children = n_children_t.view;
//     auto count_children_added = count_children_added_t.view;

//     auto count_n_children = ([=] TMOL_DEVICE_FUNC(int i) {
//         T parent = parents[i];
//         if (i != parent) {
//             accummulate<D, T>::add(n_children[parent], 1);
//         }
//     });
//     DeviceDispatch<D>::forall(n_kintree_nodes, count_n_children);
//     DeviceDispatch<D>::scan(n_children.data(), child_list_span.data(),
//     n_kintree_nodes + 1, mgpu::plus<T>());

//     auto fill_child_list = ([=] TMOL_DEVICE_FUNC(int i) {
//         T parent = parents[i];
//         T child_list_start = child_list_span[parent];
//         T my_offset = accummulate<D, T>::add(count_children_added[parent],
//         1); child_list[child_list_start + my_offset] = i;
//     });
//     DeviceDispatch<D>::forall(n_kintree_nodes, fill_child_list);

//     auto print_child_list = ([=] TMOL_DEVICE_FUNC(int i) {
//         T start = child_list_span[i];
//         T end = child_list_span[i + 1];
//         printf("Node %d, with span (%d to %d), has children: ", i, start,
//         end); for (T j = start; j < end; ++j) {
//             printf("%d ", child_list[j]);
//         }
//         printf("\n");
//     });
//     DeviceDispatch<D>::forall(n_kintree_nodes, print_child_list);

// }

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

  printf("get_n_atoms_for_block %d %d\n", n_poses, max_n_blocks);
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, get_n_atoms_for_block);
  printf("scan_and_return_total\n");
  Int n_kfo_atoms =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          block_n_atoms.data(),
          block_kfo_offset.data(),
          n_poses * max_n_blocks,
          mgpu::plus_t<Int>());
  printf("n_kfo_atoms %d\n", n_kfo_atoms);

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
    printf("get_kfo_mapping %d %d %d %d\n", pose, block, atom, block_type);

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
  printf("get_kfo_mapping %d\n", max_n_atoms_per_block);
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
template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Int>
auto KinForestFromStencil<DeviceDispatch, D, Int>::get_kfo_atom_parents(
    TView<Int, 2, D> pose_stack_block_type,                 // P x L
    TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
    TView<Int, 2, D> pose_stack_ff_parent,                  // P x L
    TView<Int, 2, D> pose_stack_ff_conn_to_parent,          // P x L
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
    int const conn_to_parent = pose_stack_ff_conn_to_parent[pose][block];
    int const ff_in = pose_stack_block_in_and_first_out[pose][block][0];

    int const bt_parent_for_atom =
        block_type_parents[block_type][conn_to_parent][atom];
    printf(
        "pose %d block %d atom %d block_type %d conn_to_parent %d ff_in %d "
        "bt_parent_for_atom %d\n",
        pose,
        block,
        atom,
        block_type,
        conn_to_parent,
        ff_in,
        bt_parent_for_atom);
    if (bt_parent_for_atom < 0) {
      // Inter-residue connection
      int const parent_block = pose_stack_ff_parent[pose][block];
      printf("parent_block %d\n", parent_block);
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
          printf("parent_block_type %d\n", parent_block_type);
          int const parent_conn =
              pose_stack_inter_residue_connections[pose][block][conn_to_parent]
                                                  [1];
          printf("parent_conn %d\n", parent_conn);
          int const parent_conn_atom =
              block_type_conn_atom[parent_block_type][parent_conn];
          printf("parent_conn_atom %d\n", parent_conn_atom);
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
    TView<Int, 2, D> pose_stack_block_type,         // x
    TView<Int, 2, D> pose_stack_ff_conn_to_parent,  // x
    TView<Int, 2, D> kfo_2_orig_mapping,            // x
    TView<Int, 1, D> kfo_parent_atoms,              // x
    TView<Int, 1, D> block_type_n_conn              // x
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
      accumulate<D, Int>::add(n_jump_children[parent], Int(1));
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
        int const conn_to_parent = pose_stack_ff_conn_to_parent[pose][block];
        // printf("n_conn %d conn_to_parent %d\n", n_conn, conn_to_parent);
        if (conn_to_parent == n_conn) {
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

// static auto EIGEN_DEVICE_FUNC get_c1_and_c2_atoms(
//     int jump_atom,
//     TView<Int, 1, D> atom_is_jump,
//     TView<Int, 2, D> child_list_span,
//     TView<Int, 1, D> child_list,
//     TView<Int, 1, D> parents) -> tuple {
//   int first_nonjump_child = -1;
//   int second_nonjump_child = -1;
//   for (int child_ind = child_list_span[jump_atom][0];
//        child_ind < child_list_span[jump_atom][1]; ++child_ind) {
//     int child_atom = child_list[child_ind];
//     if (atom_is_jump[child_atom]) {
//       continue;
//     }
//     if (first_nonjump_child == -1) {
//       first_nonjump_child = child_atom;
//     } else {
//       second_nonjump_child = child_atom;
//       break;
//     }
//   }
//   if (first_nonjump_child == -1) {
//     int jump_parent = parents[jump_atom];
//     assert(jump_parent != jump_atom);
//     return get_c1_and_c2_atoms(jump_parent, atom_is_jump, child_list_span,
//                                child_list, parents);
//   }
//   for (int grandchild_ind = child_list_span[first_nonjump_child][0];
//        grandchild_ind < child_list_span[first_nonjump_child][1];
//        ++grandchild_ind) {
//     int grandchild_atom = child_list[grandchild_ind];
//     if (!atom_is_jump[grandchild_atom]) {
//       return std::make_tuple(first_nonjump_child, grandchild_atom);
//     }
//   }
//   if (second_nonjump_child == -1) {
//     int jump_parent = parents[jump_atom];
//     assert(jump_parent != jump_atom);
//     return get_c1_and_c2_atoms(jump_parent, atom_is_jump, child_list_span,
//                                child_list, parents);
//   }
//   return std::make_tuple(first_nonjump_child, second_nonjump_child);
// }

// }

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
