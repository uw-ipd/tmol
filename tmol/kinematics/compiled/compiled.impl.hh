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

// @numba.jit(nopython=True)
// def stub_defined_for_jump_atom(jump_atom, atom_is_jump, child_list_span,
// child_list):
//     #  have to handle a couple of cases here:
//     #
//     #  note -- in counting dependent atoms, exclude JumpAtom's
//     #
//     #
//     #  1. no dependent atoms --> no way to define new coord sys
//     #     on this end. ergo take parent's M and my xyz
//     #
//     #  2. one dependent atom --> no way to define unique coord
//     #     on this end, still take parent's M and my xyz
//     #
//     #  3. two or more dependent atoms
//     #     a) if my first atom has a dependent atom, use
//     #        myself, my first atom, and his first atom
//     #
//     #     b) otherwise, use
//     #        myself, my first atom, my second atom

//     first_nonjump_child = -1
//     for child_ind in range(
//         child_list_span[jump_atom, 0], child_list_span[jump_atom, 1]
//     ):
//         child_atom = child_list[child_ind]
//         if atom_is_jump[child_atom]:
//             continue
//         if first_nonjump_child == -1:
//             first_nonjump_child = child_atom
//         else:
//             return True
//     if first_nonjump_child != -1:
//         for grandchild_ind in range(
//             child_list_span[first_nonjump_child, 0],
//             child_list_span[first_nonjump_child, 1],
//         ):
//             if not atom_is_jump[child_list[grandchild_ind]]:
//                 return True
//     return False

// @numba.jit(nopython=True)
// def fix_jump_nodes(
//     parents: NDArray[int][:],
//     frame_x: NDArray[int][:],
//     frame_y: NDArray[int][:],
//     frame_z: NDArray[int][:],
//     roots: NDArray[int][:],
//     jumps: NDArray[int][:],
// ):
//     # nelts = parents.shape[0]
//     n_children, child_list_span, child_list = get_children(parents)

//     atom_is_jump = numpy.full(parents.shape, 0, dtype=numpy.int32)
//     atom_is_jump[roots] = 1
//     atom_is_jump[jumps] = 1

//     for root in roots:
//         assert stub_defined_for_jump_atom(
//             root, atom_is_jump, child_list_span, child_list
//         )

//         root_c1, second_descendent = get_c1_and_c2_atoms(
//             root, atom_is_jump, child_list_span, child_list, parents
//         )

//         # set the frame_x, _y, and _z to the same values for both the root
//         # and the root's first child

//         frame_x[root] = root_c1
//         frame_y[root] = root
//         frame_z[root] = second_descendent

//         frame_x[root_c1] = root_c1
//         frame_y[root_c1] = root
//         frame_z[root_c1] = second_descendent

//         # all the other children of the root need an updated kinematic
//         description for child_ind in range(child_list_span[root, 0] + 1,
//         child_list_span[root, 1]):
//             child = child_list[child_ind]
//             if atom_is_jump[child]:
//                 continue
//             if child == root_c1:
//                 continue
//             frame_x[child] = child
//             frame_y[child] = root
//             frame_z[child] = root_c1

//     for jump in jumps:
//         if stub_defined_for_jump_atom(jump, atom_is_jump, child_list_span,
//         child_list):
//             jump_c1, jump_c2 = get_c1_and_c2_atoms(
//                 jump, atom_is_jump, child_list_span, child_list, parents
//             )

//             # set the frame_x, _y, and _z to the same values for both the
//             jump # and the jump's first child

//             frame_x[jump] = jump_c1
//             frame_y[jump] = jump
//             frame_z[jump] = jump_c2

//             frame_x[jump_c1] = jump_c1
//             frame_y[jump_c1] = jump
//             frame_z[jump_c1] = jump_c2

//             # all the other children of the jump need an updated kinematic
//             description for child_ind in range(
//                 child_list_span[jump, 0] + 1, child_list_span[jump, 1]
//             ):
//                 child = child_list[child_ind]
//                 if atom_is_jump[child]:
//                     continue
//                 if child == jump_c1:
//                     continue
//                 frame_x[child] = child
//                 frame_y[child] = jump
//                 frame_z[child] = jump_c1
//         else:
//             # ok, so... I don't understand the atom tree well enough to
//             understand this # situation. If the jump has no non-jump
//             children, then certainly none # of them need their frame
//             definitions updated c1, c2 = get_c1_and_c2_atoms(
//                 parents[jump], atom_is_jump, child_list_span, child_list,
//                 parents
//             )

//             frame_x[jump] = c1
//             frame_y[jump] = jump
//             frame_z[jump] = c2

//             # the jump may have one child; it's not entirely clear to me
//             # what frame the child should have!
//             # TO DO: figure this out
//             for child_ind in range(
//                 child_list_span[jump, 0] + 1, child_list_span[jump, 1]
//             ):
//                 child = child_list[child_ind]
//                 if atom_is_jump[child]:
//                     continue
//                 frame_x[child] = c1
//                 frame_y[child] = jump
//                 frame_z[child] = c2

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
    printf("first_pass_frame_xyz %d %d\n", i, parent);
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
      printf(
          "get_c1_and_c2_atoms: jump atom %d, %d, %d\n",
          jump_atom,
          first_nonjump_child,
          second_nonjump_child);
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
        printf("c1 c2 %d %d\n", c1, c2);

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
          printf("c1 c2 %d %d\n", c1, c2);

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
// S = maximum number of scan paths in any generation in any block type
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
    TView<Int, 5, D> block_type_nodes_for_gens,   // y - T x I x O x G x N
    TView<Int, 5, D> block_type_scan_path_starts  // y - T x I x O x G x S
    )
    -> std::tuple<
        TPack<Int, 2, Device::CPU>,  // dfs_order_of_ff_edges_t
        TPack<Int, 1, Device::CPU>,  // n_ff_edges_t
        TPack<Int, 2, Device::CPU>,  // first_ff_edge_for_block_cpu_t
        TPack<Int, 2, Device::CPU>,  // max_n_gens_for_ff_edge_cpu_t
        TPack<Int, 2, Device::CPU>,  // first_child_of_ff_edge_t
        TPack<Int, 2, Device::CPU>,  // first_ff_edge_for_block_cpu_t
        TPack<Int, 2, Device::CPU>   // delay_for_edge_t
        > {
  // The final step is to construct the nodes, scans, and gens tensors
  // from the per-block-type stencils.
  //

  // For each block, we need to know which FoldForest edge builds it.
  // For each FF edge, we need to know its generational delay.
  // With that, we can calculate the generational delay for each block.
  // For each block-scan-path, we need to know its offset into the nodes tensor.
  // For each block-scan path, we need to know its offset into the block-scans
  // list Then we can ask each block-scan path how many nodes it has, and
  // generate the
  //   offset using scan.
  // We need to know how many block scan paths there are.
  // We need to map block-scan path index to block, generation, and
  // scan-within-the-generation.

  // In order to know the block-scan-path index for any block-scan path, we have
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
  int const max_n_res_per_pose = pose_stack_block_type.size(1);
  int const max_n_edges_per_ff = ff_edges_cpu.size(1);
  int const max_n_input_conn = block_type_kts_conn_info.size(1);
  int const max_n_output_conn = block_type_kts_conn_info.size(1);
  int const max_n_gens = block_type_nodes_for_gens.size(3);
  int const max_n_nodes_per_gen = block_type_nodes_for_gens.size(4);
  int const max_n_scan_paths_per_gen = block_type_scan_path_starts.size(4);

  // Step 1:
  printf("Step 1\n");
  // Step N-11:
  // Construct a depth-first traversal of the fold-forest edges to determine a
  // partial order (and incidental total order) of the edges in the fold forest.
  // Do this by inserting all edges into an edge-list representation and then
  // starting at the root.
  auto dfs_order_of_ff_edges_t =
      TPack<Int, 2, Device::CPU>::zeros({n_poses, max_n_edges_per_ff});
  auto dfs_order_of_ff_edges = dfs_order_of_ff_edges_t.view;
  auto n_ff_edges_t =
      TPack<Int, 1, Device::CPU>::full({n_poses}, max_n_edges_per_ff);
  auto n_ff_edges = n_ff_edges_t.view;
  std::vector<std::vector<std::list<std::tuple<int, int>>>> ff_children(
      n_poses);
  std::vector<std::vector<bool>> has_parent(n_poses);
  for (int pose = 0; pose < n_poses; ++pose) {
    ff_children[pose].resize(max_n_res_per_pose);
    has_parent[pose].resize(max_n_res_per_pose, false);
  }
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int edge = 0; edge < max_n_edges_per_ff; ++edge) {
      int const ff_edge_type = ff_edges_cpu[pose][edge][0];
      printf("ff_edge_type %d %d %d\n", pose, edge, ff_edge_type);
      if (ff_edge_type == -1) {
        n_ff_edges[pose] =
            edge;  // we are one past the last edge, thus at the number of edges
        continue;
      }
      int const ff_edge_start = ff_edges_cpu[pose][edge][1];
      int const ff_edge_end = ff_edges_cpu[pose][edge][2];
      printf(
          "%d %d %d %d %d\n",
          pose,
          edge,
          ff_edge_type,
          ff_edge_start,
          ff_edge_end);
      has_parent[pose][ff_edge_end] = true;
      ff_children[pose][ff_edge_start].push_back(
          std::make_tuple(ff_edge_end, edge));
    }
  }
  // deduce root block
  // There is an implicit jump edge from the virtual root of the kinforest to
  // the root of each pose's fold tree. It is okay for multiple edges to come
  // out of the root block and so we talk about the root block and not the root
  // edge.
  std::vector<int> root_block(n_poses, -1);
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int block = 0; block < max_n_res_per_pose; ++block) {
      if (!ff_children[pose][block].empty() && !has_parent[pose][block]) {
        if (root_block[pose] != -1) {
          throw std::runtime_error("Multiple root blocks in fold tree");
        }
        root_block[pose] = block;
        printf("root_block %d %d\n", pose, block);
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
      std::tuple<int, int> const child_edge_tuple = stack.back();
      stack.pop_back();
      int const block = std::get<0>(child_edge_tuple);
      int const edge = std::get<1>(child_edge_tuple);
      printf(
          "dfs %d %d: e %d (%d %d)\n",
          pose,
          count_dfs_ind,
          edge,
          ff_edges_cpu[pose][edge][1],
          ff_edges_cpu[pose][edge][2]);
      dfs_order_of_ff_edges[pose][count_dfs_ind] = edge;
      count_dfs_ind += 1;
      for (auto const& child : ff_children[pose][block]) {
        stack.push_back(child);
      }
    }
  }

  // Step 2:
  printf("Step 2\n");
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
      TPack<Int, 2, Device::CPU>::full({n_poses, max_n_res_per_pose}, -1);
  auto first_ff_edge_for_block_cpu = first_ff_edge_for_block_cpu_t.view;
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
      // int max_n_gens = 0;
      if (ff_edge_type == 0) {
        int const increment = (ff_edge_start < ff_edge_end) ? 1 : -1;
        int const stop = ff_edge_end + increment;
        for (int block = ff_edge_start + increment; block != stop;
             block += increment) {
          first_ff_edge_for_block_cpu[pose][block] = edge;
          // danger! lives on device -- int const block_type =
          // pose_stack_block_type[pose][block];
        }
      } else if (ff_edge_type == 1) {
        // jump edge! The first block is not built by the jump,
        // but the second block is.
        first_ff_edge_for_block_cpu[pose][ff_edge_end] = edge;
      }
    }
  }

  // Step 3:
  printf("Step 3\n");
  // Step N-9:
  // Find the maximum number of generations of any block type of any edge in the
  // fold forest. TEMP!!!
  auto max_n_gens_for_ff_edge_t = TPack<Int, 2, Device::CPU>::full(
      {n_poses, max_n_edges_per_ff}, max_n_gens);
  auto max_n_gens_for_ff_edge = max_n_gens_for_ff_edge_t.view;

  // Step 4:
  printf("Step 4\n");
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
      printf(
          "reverse traversal of ff edge %d %d %d %d\n",
          pose,
          edge,
          ff_edge_start,
          ff_edge_end);

      int const ff_edge_max_n_gens = max_n_gens_for_ff_edge[pose][edge];
      int max_child_gen_depth = -1;
      int second_max_child_gen_depth = -1;
      int first_child = -1;
      for (auto const& child : ff_children[pose][ff_edge_end]) {
        int const child_edge = std::get<1>(child);
        int const child_gen_depth = max_gen_depth_of_ff_edge[pose][child_edge];
        printf(
            "Looking at child of res %d: %d %d, max_child_gen_depth %d second "
            "max %d\n",
            ff_edge_end,
            child_edge,
            child_gen_depth,
            max_child_gen_depth,
            second_max_child_gen_depth);
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
      printf(
          "max_gen_depth_of_ff_edge %d %d = %d\n", pose, edge, edge_gen_depth);
      max_gen_depth_of_ff_edge[pose][edge] = edge_gen_depth;
    }
  }

  // Step 5:
  printf("Step 5\n");
  // Step N-7:
  // Compute the delay for each edge given the path decomposition of the
  // fold-forest.
  for (int pose = 0; pose < n_poses; ++pose) {
    // Now select the first edge to be built from the root block
    // and set the delay for all other edges to 1.
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
    for (auto const& child : ff_children[pose][root_block[pose]]) {
      int const child_edge = std::get<1>(child);
      if (child_edge == max_root_child_edge) {
        continue;
      }
      delay_for_edge[pose][child_edge] = 1;
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
          // Note that this edge is the root of its own scan path
          // int const child_edge_type = ff_edges_cpu[pose][child_edge][0];
          // if (child_edge_type == 0) {
          //   non_jump_ff_edge_rooted_at_scan_path
          // }
        }
      }
    }
  }
  return {
      dfs_order_of_ff_edges_t,
      n_ff_edges_t,
      first_ff_edge_for_block_cpu_t,
      max_gen_depth_of_ff_edge_t,
      first_child_of_ff_edge_t,
      max_gen_depth_of_ff_edge_t,
      delay_for_edge_t};
}

// // P = number of poses
// // L = length of the longest pose
// // T = number of block types
// // A = maximum number of atoms in any block type
// // C = maximum number of inter-residue connections in any block type
// // E = maximum number of edges in any one FoldTree of the FoldForest
// // I = maximum number of input connections in any block type
// // O = maximum number of output connections in any block type
// // G = maximum number of generations in any block type
// // N = maximum number of nodes in any generation in any block type
// // S = maximum number of scan paths in any generation in any block type
// template <
//     template <tmol::Device>
//     class DeviceDispatch,
//     tmol::Device D,
//     typename Int>
// auto KinForestFromStencil<DeviceDispatch, D, Int>::get_scans(
//     TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
//     TView<Int, 2, D> pose_stack_block_type,                 // P x L
//     TView<Int, 4, D> pose_stack_inter_residue_connections,  // P x L x C x 2
//     TView<Int, 3, CPU> ff_edges_cpu,                        // P x E x 4 --
//     0: type, 1: start, 2: stop, 3: jump ind TView<Int, 3, D> ff_edges, // P x
//     E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind TView<Int, 2, D>
//     pose_stack_ff_parent,                  // P x L TView<Int, 2, D>
//     pose_stack_ff_conn_to_parent,          // P x L TView<Int, 3, D>
//     pose_stack_block_in_and_first_out,     // P x L x 2 TView<Int, 3, D>
//     block_type_parents,                    // T x O x A TView<Int, 2, D>
//     kfo_2_orig_mapping,                    // K x 3 TView<Int, 3, D>
//     atom_kfo_index,                        // P x L x A TView<Int, 1, D>
//     block_type_jump_atom,                  // T TView<Int, 1, D>
//     block_type_n_conn,                     // T TView<Int, 2, D>
//     block_type_polymeric_conn_index,       // T x 2 - 2 is for "down" and
//     "up" connections. TView<Int, 4, D> block_type_n_gens, // T x I x O
//     TVIew<Int, 5, D> block_type_kts_conn_info,              // T x I x O x C
//     x 2 -- 2 is for gen (0) and scan (1) TView<Int, 5, D>
//     block_type_nodes_for_gens,             // T x I x O x G x N TView<Int, 4,
//     D> block_type_n_scan_paths,               // T x I x O x G TView<Int, 5,
//     D> block_type_scan_path_starts,           // T x I x O x G x S
//     TView<bool, 5, D> block_type_scan_path_is_real,         // T x I x O x G
//     x S TView<bool, 5, D> block_type_scan_path_is_inter_block,  // T x I x O
//     x G x S TView<Int, 5, D> block_type_scan_path_length            // T x I
//     x O x G x S
// ) -> std::tuple<TPack<Int, 1, D>, TPack<Int, 1, D>> {
//     // The final step is to construct the nodes, scans, and gens tensors
//     // from the per-block-type stencils.
//     //
//
//     // For each block, we need to know which FoldForest edge builds it.
//     // For each FF edge, we need to know its generational delay.
//     // With that, we can calculate the generational delay for each block.
//     // For each block-scan-path, we need to know its offset into the nodes
//     tensor.
//     // For each block-scan path, we need to know its offset into the
//     block-scans list
//     // Then we can ask each block-scan path how many nodes it has, and
//     generate the
//     //   offset using scan.
//     // We need to know how many block scan paths there are.
//     // We need to map block-scan path index to block, generation, and
//     scan-within-the-generation.
//
//     // In order to know the block-scan-path index for any block-scan path, we
//     have to
//     // count the number of block-scan paths that come before it. This can be
//     tricky
//     // because some block-scan paths continue into other blocks, and we do
//     not know
//     // a priori how many block-scan paths there are downstream of such a
//     block-scan path.
//     // For each (inter-block) scan path, we have to calculate how many
//     block-scan paths
//     // comprise it. Each scan path can be readily identified from the fold
//     forest.
//     // Each block type should identify which scan paths are inter-block so
//     it's easy to
//     // figure out for each block-scan path extend into other blocks: not all
//     do.
//
//     // Step N-5:
//
//     // Step N-4: count the number of blocks that build each
//     (perhaps-multi-res) scan path.
//
//     // Step N-3: perform a segmented scan on the number of blocks that build
//     each
//     // (perhaps-multi-res) scan path.
//
//     // Step N-2: write the number of atoms in each scan path to the
//     appropriate place
//     // in the n_atoms_for_scan_path_for_gen tensor.
//
//     // Step N-1: perform a scan on the number of atoms in each scan path to
//     get the
//     // nodes tensor offset.
//
//     // Step N: copy the scan path stencils into the nodes tensor, adding the
//     // pose-stack- and block- offsets to the atom indices. Note that the
//     upstream
//     // jump atom must be added for jump edges that are the roots of paths.
//
//     int const n_poses = pose_stack_block_type.size(0);
//     int const max_n_res_per_pose = pose_stack_block_type.size(1);
//     int const max_n_edges_per_ff = ff_edges.size(1);
//     int const max_n_input_conn = block_type_kts_conn_info.size(1);
//     int const max_n_output_conn = block_type_kts_conn_info.size(1);
//     int const max_n_gens = block_type_nodes_for_gens.size(3);
//     int const max_n_nodes_per_gen = block_type_nodes_for_gens.size(4);
//     int const max_n_scan_paths_per_gen = block_type_scan_path_starts.size(4);
//
//     auto n_sps_for_ffedge_for_gen_by_topo_sort_t = TPack<Int, 2,
//     D>::zeros({max_n_gens, n_poses * max_n_edges_per_ff}); auto
//     sp_offset_for_ffedge_for_gen_by_topo_sort_t = TPack<Int, 2,
//     D>::zeros({max_n_gens, n_poses * max_n_edges_per_ff});
//
//     // Step 1:
//     // Step N-11:
//     // Construct a depth-first traversal of the fold-forest edges to
//     determine a
//     // partial order (and incidental total order) of the edges in the fold
//     forest.
//     // Do this by inserting all edges into an edge-list representation and
//     then
//     // starting at the root.
//     auto dfs_order_of_ff_edges_t = TPack<Int, 2,
//     Device::CPU>::zeros({n_poses, max_n_edges_per_ff}); auto
//     dfs_order_of_ff_edges = dfs_order_of_ff_edges_t.view; auto n_ff_edges_t =
//     TPack<Int, 1, Device::CPU>::zeros({n_poses}); auto n_ff_edges =
//     n_ff_edges_t.view; std::vector<std::vector<std::list<std::tuple<int, int>
//     > > ff_children(n_poses); std::vector<std::vector<bool> >
//     has_parent(n_poses); for (int pose = 0; pose < n_poses; ++pose) {
//       ff_children[pose].resize(max_n_res_per_pose);
//       has_parent[pose].resize(max_n_res_per_pose, false);
//     }
//     for (int pose = 0; pose < n_poses; ++pose) {
//       for (int edge = 0; edge < max_n_edges_per_ff; ++edge) {
//         int const ff_edge_type = ff_edges_cpu[pose][edge][0];
//         if (ff_edge_type == -1) {
//           n_ff_edges[pose] = edge; // we are one past the last edge, thus at
//           the number of edges continue;
//         }
//         int const ff_edge_start = ff_edges_cpu[pose][edge][1];
//         int const ff_edge_end = ff_edges_cpu[pose][edge][2];
//         has_parent[pose][ff_edge_end] = true;
//         ff_children[pose][ff_edge_start].push_back(std::make_tuple(ff_edge_end,
//         edge));
//       }
//     }
//     // deduce root block
//     // There is an implicit jump edge from the virtual root of the kinforest
//     to the
//     // root of each pose's fold tree. It is okay for multiple edges to come
//     out of
//     // the root block and so we talk about the root block and not the root
//     edge. std::vector<int> root_block(n_poses, -1); for (int pose = 0; pose <
//     n_poses; ++pose) {
//       for (int block = 0; block < max_n_res_per_pose; ++block) {
//         if (!ff_children[pose][block].empty() && !has_parent[pose][block]) {
//           if (root_block[pose] != -1) {
//             throw std::runtime_error("Multiple root blocks in fold tree");
//           }
//           root_block[pose] = block;
//         }
//       }
//     }
//     // Now let's perform the depth-first traversals from each pose.
//     for (int pose = 0; pose < n_poses; ++pose) {
//       int count_dfs_ind = 0;
//       std::vector<std::tuple<int, int>> stack;
//       for (auto const& child : ff_children[pose][root_block[pose]]) {
//         stack.push_back(child);
//       }
//       while (!stack.empty()) {
//         std::tuple<int, int> const child = stack.back();
//         stack.pop_back();
//
//         dfs_order_of_ff_edges[pose][count_dfs_ind].push_back(std::get<1>(child));
//         count_dfs_ind += 1;
//         for (auto const& child : ff_children[pose][block]) {
//           stack.push_back(child);
//         }
//       }
//     }
//
//     // Step 2:
//     // Step N-10:
//     // Write down for each residue the first edge in the fold forest that
//     builds it
//     // using the partial order of the fold-forest edges. Note that an edge's
//     start
//     // residue is not first built by that edge.
//     // In the same traversal,
//     // let's also calculate the maximum number of generations of any block
//     type
//     // of any edge?????
//     // OR let's just assume that every edge has the same number of
//     generations
//     // for now and TO DO: write a segmented scan on max() to identify the
//     number
//     // of generations for each particular residue that is built by an edge.
//     auto first_ff_edge_for_block_cpu_t = TPack<Int, 2,
//     Device::CPU>::full({n_poses, max_n_res_per_pose}, -1); auto
//     first_ff_edge_for_block_cpu = first_ff_edge_for_block_cpu_t.view; auto
//     max_n_gens_for_ff_edge_cpu_t = TPack<Int, 2,
//     Device::CPU>::zeros({n_poses, max_n_edges_per_ff}); auto
//     max_n_gens_for_ff_edge_cpu = max_n_gens_for_ff_edge_cpu_t.view; for (int
//     pose = 0; pose < n_poses; ++pose) {
//
//       for (int edge_dfs_ind = 0; edge_dfs_ind < max_n_edges_per_ff;
//       ++edge_dfs_ind) {
//         int const edge = dfs_order_of_ff_edges[pose][edge_dfs_ind];
//         if (edge == -1) {
//
//           break;
//         }
//         int const ff_edge_type = ff_edges_cpu[pose][edge][0];
//         int const ff_edge_start = ff_edges_cpu[pose][edge][1];
//         int const ff_edge_end = ff_edges_cpu[pose][edge][2];
//         // int max_n_gens = 0;
//         if (ff_edge_type == 0) {
//           int const increment = (ff_edge_start < ff_edge_end) ? 1 : -1;
//           int const stop = ff_edge_end + increment;
//           for (int block = ff_edge_start + increment; block != stop; block +=
//           increment) {
//             first_ff_edge_for_block_cpu[pose][block] = edge;
//             // danger! lives on device -- int const block_type =
//             pose_stack_block_type[pose][block];
//           }
//         }
//       }
//     }
//
//     // Step 3:
//     // Step N-9:
//     // Find the maximum number of generations of any block type of any edge
//     in the fold forest.
//     // TEMP!!!
//     auto max_n_gens_for_ff_edge_t = TPack<Int, 1, Device::CPU>::full({n_poses
//     * max_n_edges_per_ff}, max_n_gens);
//
//     // Step 4:
//     // Step N-8:
//     // Decompose the fold-forest into paths, minimizing the maximu number of
//     generations.
//     // Determine the generational delay of each edge.
//     // Then determine the input and output connections for each block. <-- Do
//     on GPU, entirely parallelizable. auto first_child_of_ff_edge_t =
//     TPack<Int, 2, Device::CPU>::full({n_poses, max_n_edges_per_ff}, -1); auto
//     max_gen_depth_of_ff_edge_t = TPack<Int, 2, Device::CPU>::zeros({n_poses,
//     max_n_edges_per_ff}); auto delay_for_edge_t = TPack<Int, 2,
//     Device::CPU>::zeros({n_poses, max_n_edges_per_ff}); auto
//     first_child_of_ff_edge = first_child_of_ff_edge_t.view; auto
//     max_gen_depth_of_ff_edge = max_gen_depth_of_ff_edge_t.view; auto
//     delay_for_edge = delay_for_edge_t.view; for (int pose = 0; pose <
//     n_poses; ++pose) {
//       // traverse edges in reverse order
//       for (int edge_in_dfs_ind = n_ff_edges[pose] - 1; edge_in_dfs_ind >= 0;
//       edge_in_dfs_ind--) {
//         int const edge = dfs_order_of_ff_edges[pose][edge_in_dfs_ind];
//         int const ff_edge_type = ff_edges_cpu[pose][edge][0];
//         int const ff_edge_start = ff_edges_cpu[pose][edge][1];
//         int const ff_edge_end = ff_edges_cpu[pose][edge][2];
//
//         int max_child_gen_depth = -1;
//         int first_child = -1;
//         for (auto const & child: ff_children[pose][ff_edge_end]) {
//           int const child_edge = std::get<1>(child);
//           int const child_gen_depth =
//           max_gen_depth_of_ff_edge[pose][child_edge]; if (child_gen_depth >
//           max_child_gen_depth) {
//             max_child_gen_depth = child_gen_depth;
//             first_child = child_edge;
//           }
//         }
//         first_child_of_ff_edge[pose][edge] = first_child;
//       }
//     }
//
//     // Step 5:
//     // Step N-7:
//     // Compute the delay for each edge given the path decomposition of the
//     fold-forest. for (int pose = 0; pose < n_poses; ++pose) {
//
//       // Now select the first edge to be built from the root block
//       // and set the delay for all other edges to 1.
//       int max_root_child_gen_depth = -1;
//       int max_root_child_edge = -1;
//       for (auto const & child: ff_children[pose][root_block[pose]]) {
//         int const child_edge = std::get<1>(child);
//         int const child_gen_depth =
//         max_gen_depth_of_ff_edge[pose][child_edge]; if (child_gen_depth >
//         max_root_child_gen_depth) {
//           max_root_child_gen_depth = child_gen_depth;
//           max_root_child_edge = child_edge;
//         }
//       }
//       edge_delay[pose][max_root_child_edge] = 0;
//       for (auto const & child: ff_children[pose][root_block[pose]]) {
//         int const child_edge = std::get<1>(child);
//         if (child_edge == max_root_child_edge) {
//           continue;
//         }
//         edge_delay[pose][child_edge] = 1;
//       }
//
//       for (int edge_in_dfs_ind = 0; edge_in_dfs_ind < n_ff_edges[pose];
//       ++edge_in_dfs_ind) {
//         int const edge = dfs_order_of_ff_edges[pose][edge_in_dfs_ind];
//         int const ff_edge_type = ff_edges_cpu[pose][edge][0];
//         int const ff_edge_start = ff_edges_cpu[pose][edge][1];
//         int const ff_edge_end = ff_edges_cpu[pose][edge][2];
//         int const first_child = first_child_of_ff_edge[pose][edge];
//         int const edge_delay = delay_for_edge[pose][edge];
//         for (auto const & child: ff_children[pose][ff_edge_end]) {
//           int const child_edge = std::get<1>(child);
//           if (child_edge == first_child) {
//             edge_delay[pose][child_edge] = edge_delay;
//           } else {
//             edge_delay[pose][child_edge] = edge_delay + 1;
//             // Note that this edge is the root of its own scan path
//             int const child_edge_type = ff_edges_cpu[pose][child_edge][0];
//             if (child_edge_type == 0) {
//               non_jump_ff_edge_rooted_at_scan_path
//             }
//           }
//         }
//       }
//     }
//     // Step 6
//     // Step N-6:
//     // Construct a topological sort of the fold-forest edges.
//     // The sorting is done by edge delay first and then by depth
//     // within the tree second. E.g. the edge (0,1) < (1,0)
//     // and (0,1) < (0,2) and (0,2) < (1,1)
//
//
//     // Step 7
//     // Step N-5:
//     // Mark the scan paths that root each non-jump fold-forest edge
//     // This will store the global indexing of the fold-forest edge rather
//     // than the per-pose indexing, but they can be interconverted easily:
//     // pose_ff_edge_index = global_edge_index % max_n_edges_per_ff
//     auto non_jump_ff_edge_rooted_at_scan_path_t = TPack<Int, 3, D>::full(
//       {n_poses, max_n_res_per_pose, max_n_gens, max_n_scan_paths_per_gen}, -1
//     );
//     auto non_jump_ff_edge_rooted_at_scan_path =
//     non_jump_ff_edge_rooted_at_scan_path_t.view; auto
//     mark_scan_paths_that_root_non_jum_fold_forest_edges = ([=]
//     TMOL_DEVICE_FUNC (int i){
//       int const pose = i / max_n_edges_per_ff;
//       int const edge = i % max_n_edges_per_ff;
//       int const ff_edge_type = ff_edges[pose][edge][0];
//       if (ff_edge_type == 1 || ff_edge_type == -1) {
//         // Jump edge or sentinel marking non-edge.
//         return;
//       }
//       int const ff_edge_start = ff_edges[pose][edge][1];
//       int const ff_edge_end = ff_edges[pose][edge][2];
//       int const start_block_type =
//       pose_stack_block_type[pose][ff_edge_start]; int const start_block_in =
//       pose_stack_block_in_and_first_out[pose][ff_edge_start][0]; int const
//       start_block_out =
//       pose_stack_block_in_and_first_out[pose][ff_edge_start][1]; int const
//       start_block_type_out_conn_ind =
//       block_type_polymeric_conn_atom[start_block_type][(ff_edge_start <
//       ff_edge_end) ? 1 : 0];
//
//       int const exitting_scan_path_gen =
//       block_type_kts_conn_info[start_block_type][start_block_in][start_block_out][start_block_type_out_conn_ind][0];
//       int const exitting_scan_path =
//       block_type_kts_conn_info[start_block_type][start_block_in][start_block_out][start_block_type_out_conn_ind][1];
//       non_jump_ff_edge_rooted_at_scan_path[pose][ff_edge_start][exitting_scan_path_gen][exitting_scan_path]
//       = (
//         pose * max_n_edges_per_ff + edge
//       );
//     });
//     DeviceDispatch<D>::template forall<launch_t>(n_poses *
//     max_n_edges_per_ff, mark_scan_paths_that_root_non_jum_fold_forest_edges);
//
//     // Step 8
//     // Step N-4:
//     // Count the number of single-block-scan-paths that build each ff-edge
//     for each generation. auto count_n_segs_for_ffedge_for_gen_by_topo_sort =
//     ([=] TMOL_DEVICE_FUNC (int i){
//         int const pose = i / (max_n_res * max_n_gens *
//         max_n_scan_paths_per_gen); i = i - pose * max_n_res * max_n_gens *
//         max_n_scan_paths_per_gen; int const block = i / (max_n_gens *
//         max_n_scan_paths_per_gen); i = i - block * max_n_gens *
//         max_n_scan_paths_per_gen; int const gen = i /
//         max_n_scan_paths_per_gen; if (i < max_n_gens) {
//             // Need indices of the start of each segment for each gen for
//             seg-scan. n_sps_for_ffedge_for_gen_segment_starts[i] = i *
//             n_poses * max_n_edges_per_ff;
//         }
//
//         int const scan_path = i % max_n_scan_paths_per_gen;
//         int const block_type = pose_stack_block_type[pose][block];
//         if (block_type == -1) { return; }
//         int ff_edge = first_ff_edge_for_block[pose][block];
//         int const ff_edge_rooted_at_scan_path =
//         non_jump_ff_edge_rooted_at_scan_path[pose][block][scan_path]; if
//         (ff_edge_rooted_at_scan_path != -1) {ff_edge =
//         ff_edge_rooted_at_scan_path;} int const ff_edge_delay =
//         delay_for_edge[ff_edge]; int const ff_edge_topo_sort_index =
//         topo_sort_index_for_edge[ff_edge];
//         // now we can increment the number of scan paths that build this edge
//         accumulate<D, T>::add(n_sp_for_ffedge_for_gen_by_topo_sort[gen +
//         ff_edge_delay][ff_edge_topo_sort_index], 1);
//     });
//     DeviceDispatch<D>::template forall<launch_t>(n_poses * max_n_res *
//     max_n_gens * max_n_scan_paths_per_gen,
//     count_n_segs_for_ffedge_for_gen_by_topo_sort);
//
//     // Step 9
//     // Step N-3:
//     // now, run segmented scan on n_sp_for_ffedge_for_gen_by_topo_sort to get
//     the offset for
//     // each ff edge for each gen so that we can then count the number of
//     atoms per scan path. auto sp_offset_for_ff_edge_for_gen_by_topo_sort_tp =
//     DeviceDispatch<D>::template segmented_scan<mgpu::scan_type_exc>(
//         n_sps_for_ffedge_for_gen_by_topo_sort.data(),
//         n_sps_for_ffedge_for_gen_segment_starts.data(),
//         n_poses * max_n_edges_per_ff * max_n_gens,
//         max_n_gens,
//         mgpu::plus_t<Int>(),
//         Int(0)
//     );
//     auto sp_offset_for_ff_edge_for_gen_by_topo_sort =
//     sp_offset_for_ff_edge_for_gen_by_topo_sort_tp.view;
//
//     // Step 10
//     // convenience function for determining the rank of a block within the
//     fold-forest
//     // edge that builds it.
//     auto polymer_edge_index_for_block = ([=] TMOL_DEVICE_FUNC (
//         typename TView<Int, 3, D> const & ff_edges,
//         int pose,
//         int edge_on_pose,
//         int block
//     ) -> int {
//         // For a polymer edge (peptide edge), return the index of a
//         particular block
//         // on that edge; e.g., for the edge 10->25, block 15 is at index 5,
//         and
//         // for the edge 25->10, block 24 is at index 1.
//         int const ff_start_block = ff_edges[pose][edge_on_pose][1];
//         int const ff_end_block = ff_edges[pose][edge_on_pose][2];
//         if (ff_start_block < ff_end_block) {
//             return block - ff_start_block;
//         } else {
//             return ff_end_block - block;
//         }
//     });
//
//     // Step 11
//     // Step N-2:
//     // Alright, now let's write down the number of atoms for each scan path
//     for each generation auto collect_n_atoms_for_scan_paths = ([=]
//     TMOL_DEVICE_FUNC (int i) {
//         int const pose = i / (max_n_res * max_n_gens *
//         max_n_scan_paths_per_gen); i = i - pose * max_n_res * max_n_gens *
//         max_n_scan_paths_per_gen; int const block = i / (max_n_gens *
//         max_n_scan_paths_per_gen); i = i - block * max_n_gens *
//         max_n_scan_paths_per_gen; int const gen = i /
//         max_n_scan_paths_per_gen;
//
//         int const scan_path = i % max_n_scan_paths_per_gen;
//         int const block_type = pose_stack_block_type[pose][block];
//         if (block_type == -1) { return; }
//         int const input_conn =
//         pose_stack_block_in_and_first_out[pose][block][0]; int const
//         first_out_conn = pose_stack_block_in_and_first_out[pose][block][1];
//
//         int ff_edge = first_ff_edge_for_block[pose][block];
//         int ff_edge_on_pose = ff_edge % n_poses;
//         int const ff_edge_rooted_at_scan_path =
//         non_jump_ff_edge_rooted_at_scan_path[pose][block][scan_path];
//
//         int extra_atom_count = 0;
//         if (ff_edge_rooted_at_scan_path != -1) {
//             ff_edge = ff_edge_rooted_at_scan_path;
//             ff_edge_on_pose = ff_edge % n_poses;
//             if (ff_edges[pose][ff_edge_on_pose][0] == 1) {
//                 // Jump edge that's rooted at this scan path. For this
//                 // edge we must add an extra atom representing the
//                 // upstream jump atom: it will not be listed as one
//                 // of the atoms in the block-type's-scan path.
//                 extra_atom_count = 1;
//             }
//         }
//         int const ff_edge_delay = delay_for_edge[ff_edge];
//         int const ff_edge_topo_sort_index =
//         topo_sort_index_for_edge[ff_edge]; int const ff_edge_gen = gen +
//         ff_edge_delay;
//
//         int const ff_edge_gen_topo_sort_index = (ff_edge_gen) * (n_poses *
//         max_n_edges_per_ff) + ff_edge_topo_sort_index; int const
//         ff_edge_gen_scan_path_offset =
//         sp_offset_for_ff_edge_for_gen_by_topo_sort[ff_edge_gen_topo_sort_index];
//         int const block_position_on_ff_edge =
//         polymer_edge_index_for_block(ff_edges, pose, ff_edge, block); int
//         const n_atoms_for_scan_path_index = ff_edge_gen_scan_path_offset +
//         block_position_on_ff_edge;
//
//         int const n_atoms_for_scan_path =
//         block_type_scan_path_length[block_type][input_conn][first_out_conn][gen][scan_path];
//
//         // And the big assignment....
//         n_atoms_for_scan_path_for_gen[gen +
//         ff_edge_delay][n_atoms_for_scan_path_index] = n_atoms_for_scan_path +
//         extra_atom_count; // ...TADA!
//     });
//     DeviceDispatch<D>::template forall<launch_t>(n_poses * max_n_res *
//     max_n_gens * max_n_scan_paths_per_gen, collect_n_atoms_for_scan_paths);
//
//     // Step 12
//     // Step N-1:
//     // And with the number of atoms for each scan path, we can now calculate
//     the offsets auto nodes_offset_for_scan_path_for_gen_tp = TPack<Int, 1,
//     D>::zeros({max_n_gens * n_poses * max_n_res_per_pose *
//     max_n_scan_paths_per_gen}); auto nodes_offset_for_scan_path_for_gen_tp =
//     n_atoms_offset_for_scan_path_for_gen_tp.view; DeviceDispatch<D>::template
//     scan<mgpu::scan_type_exc>(
//         n_atoms_for_scan_path_for_gen.data(),
//         n_atoms_offset_for_scan_path_for_gen.data(),
//         max_n_gens * n_poses * max_n_res_per_pose * max_n_scan_paths_per_gen,
//         mgpu::plus_t<Int>()
//     );
//
//     // Step 13
//     // Step N:
//     // And we can now, finally, copy the scan-path stencils into the nodes
//     tensor auto fill_nodes_tensor_from_scan_path_stencils = ([=]
//     TMOL_DEVICE_FUNC (int i) {
//         int const pose = i / (max_n_res * max_n_gens *
//         max_n_scan_paths_per_gen); i = i - pose * max_n_res * max_n_gens *
//         max_n_scan_paths_per_gen; int const block = i / (max_n_gens *
//         max_n_scan_paths_per_gen); i = i - block * max_n_gens *
//         max_n_scan_paths_per_gen; int const gen = i /
//         max_n_scan_paths_per_gen;
//
//         int const scan_path = i % max_n_scan_paths_per_gen;
//         int const block_type = pose_stack_block_type[pose][block];
//         if (block_type == -1) { return; }
//         int const input_conn =
//         pose_stack_block_in_and_first_out[pose][block][0]; int const
//         first_out_conn = pose_stack_block_in_and_first_out[pose][block][1];
//
//         int ff_edge = first_ff_edge_for_block[pose][block];
//         int ff_edge_on_pose = ff_edge % n_poses;
//         int const ff_edge_rooted_at_scan_path =
//         non_jump_ff_edge_rooted_at_scan_path[pose][block][scan_path];
//
//         int extra_atom_count = 0;
//         if (ff_edge_rooted_at_scan_path != -1) {
//             ff_edge = ff_edge_rooted_at_scan_path;
//             ff_edge_on_pose = ff_edge % n_poses;
//             if (ff_edges[pose][ff_edge_on_pose][0] == 1) {
//                 // Jump edge that's rooted at this scan path. For this
//                 // edge we must add an extra atom representing the
//                 // upstream jump atom: it will not be listed as one
//                 // of the atoms in the block-type's-scan path.
//                 extra_atom_count = 1;
//             }
//         }
//         int const ff_edge_delay = delay_for_edge[ff_edge];
//         int const ff_edge_type = ff_edges[pose][ff_edge_on_pose][0];
//         int const ff_edge_gen = gen + ff_edge_delay;
//
//         int const ff_edge_gen_topo_sort_index = ff_edge_gen * n_poses *
//         max_n_edges_per_ff + ff_edge_topo_sort_index; int const
//         ff_edge_gen_scan_path_offset =
//         sp_offset_for_ff_edge_for_gen_by_topo_sort[ff_edge_gen_topo_sort_index];
//         int const block_position_on_ff_edge =
//         polymer_edge_index_for_block(ff_edges, pose, ff_edge, block); int
//         const n_atoms_for_scan_path_index = ff_edge_gen_scan_path_offset +
//         block_position_on_ff_edge;
//
//         int const nodes_offset_for_scan_path_for_gen =
//         nodes_offset_for_scan_path_for_gen[n_atoms_for_scan_path_index];
//
//         int const n_atoms_for_scan_path =
//         block_type_scan_path_length[block_type][input_conn][first_out_conn][gen][scan_path];
//         // NOW WE ARE READY!!!
//         for (int j = 0; j < n_atoms_for_scan_path; ++j) {
//           nodes[nodes_offset_for_scan_path_for_gen + j] = (
//             block_type_nodes_for_gens[block_type][input_conn][first_out_conn][gen][scan_path][j]
//             + pose * max_n_atoms_per_pose +
//             pose_stack_block_coord_offset[pose][block]
//           )
//         }
//     });
//
//     // auto note_ff_edge_for_block_scan_path = ([=] TMOL_DEVICE_FUNC (int i){
//     //     int const pose = i / max_n_edges_per_ff;
//     //     int const edge = i % max_n_edges_per_ff;
//     //     int const ff_start_block = ff_edges[pose][edge][0];
//     //     int const ff_end_block = ff_edges[pose][edge][1];
//     //     int const ff_edge_type = ff_edges[pose][edge][2];
//     //     if (ff_start_block == -1) {
//     //         return;
//     //     }
//     //     int const block_type =
//     pose_stack_block_type[pose][ff_start_block];
//     //     if (ff_edge_type == 0) {
//     //         // polymer edge
//     //         int conn_ind = block_type_conn_atom[block_type][ff_start_block
//     < ff_end_block ? 1 : 0];
//     //         int const gen =
//     block_type_conn_info[block_type][i_input_conn][i_first_out_conn][upper_conn][0];
//     //         int const scan =
//     block_type_conn_info[block_type][i_input_conn][i_first_out_conn][upper_conn][0];
//     //         ff_edge_for_block_scan_path[pose][ff_start_block][gen][scan] =
//     edge;
//     //     } else {
//     //         // jump edge or chemical edge ????
//     //     }
//     // });
//     // DeviceDispatch<D>::template forall<launch_t>(n_poses *
//     max_n_edges_per_ff, note_ff_edge_for_block_scan_path);
//
//     // auto record_block_scan_path_natoms = ([=] TMOL_DEVICE_FUNC (int i){
//     //     int const i_pose = block_scan_path_info[i][0];
//     //     int const i_block = block_scan_path_info[i][1];
//     //     int const i_gen = block_scan_path_info[i][2];
//     //     int const i_scan = block_scan_path_info[i][3];
//     //     int const block_type = pose_stack_block_type[i_pose][i_block];
//     //     int const i_input_conn =
//     pose_stack_block_in_and_first_out[i_pose][i_block][0];
//     //     int const i_first_out_conn =
//     pose_stack_block_in_and_first_out[i_pose][i_block][1];
//     //     int const scan_size =
//     block_type_scan_length[block_type][i_input_conn][i_first_out_conn][i_gen][i_scan];
//     //     int const scan_path_index = block_scan_path_index[i];
//     //     bool const is_inter_res_block_scan_path =
//     block_type_scan_is_inter_block[block_type][i_input_conn][i_first_out_conn][i_gen][i_scan];
//     //     if (is_inter_res_block_scan_path) {
//     //         int const ff_edge =
//     ff_edge_for_block_scan_path[i_pose][i_block][i_gen][i_scan];
//     //         if (ff_edge > 0) {
//     //             // This is an inter-residue block-scan path
//     //             block_scan_path_head[scan_path_index] = true;
//     //         }
//     //     }
//     //     block_scan_path_natoms[scan_path_index] = scan_size;
//     // });
//
//     // DeviceDispatch<D>::template forall<launch_t>(n_block_scan_paths,
//     record_block_scan_path_natoms);
//     // DeviceDispatch<D>::template segmented_scan<mgpu::scan_type_exc>(
//     //     block_scan_path_head.data(),
//     //     block_scan_path_natoms.data(),
//     //     block_scan_path_offsets.data(),
//     //     n_block_scan_paths,
//     //     mgpu::plus_t<Int>());
//
//     // // Now that we have all the offsets for the block-scans, we can write
//     // // the nodes tensor.
//     // auto write_scan_path = ([=] TMOL_DEVICE_FUNC (int i){
//     //     int const i_pose = block_scan_path_info[i][0]
//     //     int const i_block = block_scan_path_info[i][1];
//     //     int const i_gen = block_scan_path_info[i][2];
//     //     int const i_scan = block_scan_path_info[i][3];
//     //     int const i_scan_offset = block_scan_path_offsets[i];
//     //     int const block_type = pose_stack_block_type[i_pose][i_block];
//     //     int const i_input_conn =
//     pose_stack_block_in_and_first_out[i_pose][i_block][0];
//     //     int const i_first_out_conn =
//     pose_stack_block_in_and_first_out[i_pose][i_block][1];
//     //     int const scan_size =
//     block_type_scan_length[block_type][i_input_conn][i_first_out_conn][i_gen][i_scan];
//     //     int const i_scan_start =
//     block_type_scan_starts[block_type][i_input_conn][i_first_out_conn][i_gen][i_scan];
//     //     for (int j = 0; j < scan_size; ++j) {
//     //         nodes[i_scan_offset + j] =
//     block_type_nodes_for_gens[block_type][i_input_conn][i_first_out_conn][i_gen][i_scan][i_scan_start
//     + j];
//     //     }
//     // });
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
