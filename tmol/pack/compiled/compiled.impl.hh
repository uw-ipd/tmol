#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>

#include <tmol/score/common/launch_box_macros.hh>
#include <tmol/score/common/tuple.hh>
#include <tmol/score/common/diamond_macros.hh>

#include <moderngpu/operators.hxx>

#include "annealer.hh"

namespace tmol {
namespace pack {
namespace compiled {

template <
    template <tmol::Device>
    class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto InteractionGraphBuilder<DeviceDispatch, D, Real, Int>::f(
    int const chunk_size,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    TView<Int, 1, D> pose_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<int32_t, 1, D> block_ind_for_rot,
    TView<int32_t, 2, D> sparse_inds,  // if we are ever dealing w > 4B
                                       // rotamers, we are in trouble
    TView<Real, 1, D> sparse_energies) -> std::
    tuple<TPack<int64_t, 3, D>, TPack<int64_t, 1, D>, TPack<Real, 1, D> > {
  int const n_poses = n_rots_for_pose.size(0);
  int const n_rotamers = pose_for_rot.size(0);
  int const max_n_blocks = n_rots_for_block.size(1);
  int const n_sparse_entries = sparse_inds.size(1);

  assert(rot_offset_for_pose.size(0) == n_poses);
  assert(n_rots_for_block.size(0) == n_poses);
  assert(block_type_ind_for_rot.size(0) == n_rotamers);
  assert(block_ind_for_rot.size(0) == n_rotamers);
  assert(sparse_inds.size(0) == 3);
  assert(sparse_energies.size(0) == n_sparse_entries);

  auto n_chunks_for_block_tp =
      TPack<int32_t, 2, D>::zeros({n_poses, max_n_blocks});
  auto n_chunks_for_block = n_chunks_for_block_tp.view;

  LAUNCH_BOX_32;
  auto count_n_chunks_for_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / max_n_blocks;
    int const block = index % max_n_blocks;
    int const n_rots = n_rots_for_block[pose][block];
    if (n_rots != 0) {
      int const n_chunks = (n_rots - 1) / chunk_size + 1;
      n_chunks_for_block[pose][block] = n_chunks;
      // printf("n_chunks_for_block[%d][%d] == %d\n", pose, block, n_rots);
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, count_n_chunks_for_block);

  auto respair_is_adjacent_tp =
      TPack<int32_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto respair_is_adjacent = respair_is_adjacent_tp.view;

  auto note_adjacent_respairs = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    // Assert: block1 < block2
    respair_is_adjacent[pose][block1][block2] = 1;
    // if (index < 100) {
    //   printf("note_adjacent_respairs %d %d %d %d %d\n", pose, rot1, rot2,
    //   block1, block2);
    // }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_sparse_entries, note_adjacent_respairs);

  auto n_chunks_for_block_pair_tp =
      TPack<int64_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto n_chunks_for_block_pair = n_chunks_for_block_pair_tp.view;

  auto note_n_chunks_for_block_pair = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / (max_n_blocks * max_n_blocks);
    index = index - pose * max_n_blocks * max_n_blocks;
    int const block1 = index / max_n_blocks;
    int const block2 = index % max_n_blocks;

    // We don't have to worry about block1 >= block2 as those will not
    // have entries in the sparse_inds input tensors
    if (respair_is_adjacent[pose][block1][block2]) {
      int const n_chunks1 = n_chunks_for_block[pose][block1];
      int const n_chunks2 = n_chunks_for_block[pose][block2];
      int const n_chunk_pairs = n_chunks1 * n_chunks2;
      // printf("respair adjacent %d %d %d; nchunk pairs %d\n", pose, block1,
      // block2, n_chunk_pairs);
      n_chunks_for_block_pair[pose][block1][block2] = n_chunk_pairs;
      n_chunks_for_block_pair[pose][block2][block1] = n_chunk_pairs;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks * max_n_blocks, note_n_chunks_for_block_pair);

  auto chunk_pair_offset_for_block_pair_tp =
      TPack<int64_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
  auto chunk_pair_offset_for_block_pair =
      chunk_pair_offset_for_block_pair_tp.view;

  // Okay, now let's figure out which chunk pairs are near each other
  int const n_adjacent_chunk_pairs_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_chunks_for_block_pair.data(),
          chunk_pair_offset_for_block_pair.data(),
          n_poses * max_n_blocks * max_n_blocks,
          mgpu::plus_t<Int>());

  auto chunk_pair_adjacency_tp =
      TPack<int64_t, 1, D>::zeros({n_adjacent_chunk_pairs_total});
  auto chunk_pair_adjacency = chunk_pair_adjacency_tp.view;

  auto note_adjacent_chunk_pairs = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    int const block1_rot_offset = rot_offset_for_block[pose][block1];
    int const block2_rot_offset = rot_offset_for_block[pose][block2];
    int const local_rot1 = rot1 - block1_rot_offset;
    int const local_rot2 = rot2 - block2_rot_offset;
    int const chunk1 = local_rot1 / chunk_size;
    int const chunk2 = local_rot2 / chunk_size;
    int const n_rots_block1 = n_rots_for_block[pose][block1];
    int const n_rots_block2 = n_rots_for_block[pose][block2];
    int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;

    int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
    int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
    int const chunk1_size = (overhang1 > chunk_size ? chunk_size : overhang1);
    int const chunk2_size = (overhang2 > chunk_size ? chunk_size : overhang2);

    int const block_pair_chunk_offset_ij =
        chunk_pair_offset_for_block_pair[pose][block1][block2];
    int const block_pair_chunk_offset_ji =
        chunk_pair_offset_for_block_pair[pose][block2][block1];

    // if (index < 100) {
    //   printf("pose %d rot1 %d rot2 %d block1 %d block2 %d block1_rot_offset
    //   %d block2_rot_offset %d local_rot1 %d local_rot2 %d chunk1 %d chunk2 %d
    //   n_rots_block1 %d n_rots_block2 %d n_chunks1 %d n_chunks2 %d overhang1
    //   %d overhang2 %d chunk1_size %d chunk2_size %d
    //   block_pair_chunk_offset_ij %d block_pair_chunk_offset_ji %d\n", pose,
    //   rot1, rot2, block1, block2, block1_rot_offset, block2_rot_offset,
    //   local_rot1, local_rot2, chunk1, chunk2, n_rots_block1, n_rots_block2,
    //   n_chunks1, n_chunks2, overhang1, overhang2, chunk1_size, chunk2_size,
    //   block_pair_chunk_offset_ij, block_pair_chunk_offset_ji);
    // }

    // multiple threads will write exactly these values to these entries in the
    // chunk_pair_adjacency table
    chunk_pair_adjacency
        [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2] =
            chunk1_size * chunk2_size;
    chunk_pair_adjacency
        [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1] =
            chunk1_size * chunk2_size;
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_sparse_entries, note_adjacent_chunk_pairs);

  auto chunk_pair_offsets_tp =
      TPack<int64_t, 1, D>::zeros({n_adjacent_chunk_pairs_total});
  auto chunk_pair_offsets = chunk_pair_offsets_tp.view;

  int64_t const n_two_body_energies =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          chunk_pair_adjacency.data(),
          chunk_pair_offsets.data(),
          n_adjacent_chunk_pairs_total,
          mgpu::plus_t<Int>());

  auto energy2b_tp = TPack<Real, 1, D>::zeros({n_two_body_energies});
  auto energy2b = energy2b_tp.view;

  auto record_energies_in_energy2b = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    Real const energy = sparse_energies[index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    int const block1_rot_offset = rot_offset_for_block[pose][block1];
    int const block2_rot_offset = rot_offset_for_block[pose][block2];
    int const local_rot1 = rot1 - block1_rot_offset;
    int const local_rot2 = rot2 - block2_rot_offset;
    int const chunk1 = local_rot1 / chunk_size;
    int const chunk2 = local_rot2 / chunk_size;
    int const n_rots_block1 = n_rots_for_block[pose][block1];
    int const n_rots_block2 = n_rots_for_block[pose][block2];
    int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;

    int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
    int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
    int const chunk1_size = (overhang1 > chunk_size ? chunk_size : overhang1);
    int const chunk2_size = (overhang2 > chunk_size ? chunk_size : overhang2);

    int const rot_ind_wi_chunk1 = local_rot1 - chunk1 * chunk_size;
    int const rot_ind_wi_chunk2 = local_rot2 - chunk2 * chunk_size;

    int const block_pair_chunk_offset_ij =
        chunk_pair_offset_for_block_pair[pose][block1][block2];
    int const block_pair_chunk_offset_ji =
        chunk_pair_offset_for_block_pair[pose][block2][block1];

    int const chunk_offset_ij = chunk_pair_offsets
        [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2];
    int const chunk_offset_ji = chunk_pair_offsets
        [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1];

    // if (index < 100) {
    //   printf("record: pose %d rot1 %d rot2 %d E %6.3f block1 %d block2 %d
    //   block1_rot_offset %d block2_rot_offset %d local_rot1 %d local_rot2 %d
    //   chunk1 %d chunk2 %d n_rots_block1 %d n_rots_block2 %d n_chunks1 %d
    //   n_chunks2 %d overhang1 %d overhang2 %d chunk1_size %d chunk2_size %d
    //   rot_ind_wi_chunk1 %d rot_ind_wi_chunk2 %d block_pair_chunk_offset_ij %d
    //   block_pair_chunk_offset_ji %d\n", pose, rot1, rot2, energy, block1,
    //   block2, block1_rot_offset, block2_rot_offset, local_rot1, local_rot2,
    //   chunk1, chunk2, n_rots_block1, n_rots_block2, n_chunks1, n_chunks2,
    //   overhang1, overhang2, chunk1_size, chunk2_size, rot_ind_wi_chunk1,
    //   rot_ind_wi_chunk2, block_pair_chunk_offset_ij,
    //   block_pair_chunk_offset_ji);
    // }

    energy2b
        [chunk_offset_ij + rot_ind_wi_chunk1 * chunk2_size
         + rot_ind_wi_chunk2] = energy;
    energy2b
        [chunk_offset_ji + rot_ind_wi_chunk2 * chunk1_size
         + rot_ind_wi_chunk1] = energy;
  });
  DeviceDispatch<D>::template forall<launch_t>(
      n_sparse_entries, record_energies_in_energy2b);

  // Mark the chunk_pair_offset_for_block_pair that are not adjacent w/ -1s
  // Mark the chunk_pair_offsets that are not adjacent w/ -1s

  auto sentinel_out_non_adjacent_block_pairs =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const pose = index / (max_n_blocks * max_n_blocks);
        index = index - pose * max_n_blocks * max_n_blocks;
        int const block1 = index / max_n_blocks;
        int const block2 = index % max_n_blocks;

        // We don't have to worry about block1 >= block2 as those will not
        // have entries in the sparse_inds input tensors
        if (block1 <= block2 && !respair_is_adjacent[pose][block1][block2]) {
          chunk_pair_offset_for_block_pair[pose][block1][block2] = -1;
          chunk_pair_offset_for_block_pair[pose][block2][block1] = -1;
        }
      });
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks * max_n_blocks,
      sentinel_out_non_adjacent_block_pairs);

  auto sentinel_out_non_adjacent_chunk_pairs =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const n_pairs_for_chunk = chunk_pair_adjacency.data()[index];
        if (n_pairs_for_chunk == 0) {
          // if (index < 100) {
          // 	printf("Non adjacent chunk pair %d\n", index);
          // }
          chunk_pair_offsets[index] = -1;
        }
      });
  DeviceDispatch<D>::template forall<launch_t>(
      n_adjacent_chunk_pairs_total, sentinel_out_non_adjacent_chunk_pairs);

  return std::make_tuple(
      chunk_pair_offset_for_block_pair_tp, chunk_pair_offsets_tp, energy2b_tp);
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
