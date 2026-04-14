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
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto InteractionGraphBuilder<DeviceDispatch, D, Real, Int>::f(
    int const chunk_size,
    int const max_n_block_types,
    TView<Int, 1, D> n_rots_for_pose,
    TView<Int, 1, D> rot_offset_for_pose,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    TView<Int, 1, D> pose_for_rot,
    TView<Int, 1, D> block_type_ind_for_rot,
    TView<int32_t, 1, D> block_ind_for_rot,
    TView<int32_t, 2, D>
        sparse_inds,  // why int32? well, if we are ever dealing w > 4B
                      // rotamers, we are in trouble
    TView<Real, 1, D> sparse_energies)
    -> std::tuple<
        TPack<
            int64_t,
            1,
            tmol::Device::CPU>,  // max_n_bump_checked_rotamers_per_pose
        TPack<Int, 1, D>,        // n_molten_blocks_per_pose
        TPack<Int, 1, D>,        // n_bc_rots_per_pose
        TPack<Int, 1, D>,        // bc_rot_offset_for_pose
        TPack<Int, 2, D>,        // n_bc_rots_for_molten_block
        TPack<Int, 2, D>,        // bc_rot_offset_for_molten_block
        TPack<Int, 1, D>,        // molten_block_ind_for_bc_rot
        TPack<int64_t, 2, D>,    // rotamer_for_nonmolten_block
        TPack<int64_t, 1, D>,    // bc_rot_to_orig_rot

        TPack<Real, 1, D>,
        TPack<int64_t, 3, D>,
        TPack<int64_t, 1, D>,
        TPack<Real, 1, D> > {
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

  LAUNCH_BOX_32;

  // We will perform a "bump check" to eliminate some rotamers if they
  // are too high in energy to be worth considering; we do this after
  // first constructing a first-pass version of the interaction graph
  // So we declare this tensor outside of the scope of the
  // first-pass construction.
  auto keep_rotamer_tp = TPack<int64_t, 1, D>::zeros({n_rotamers});
  auto keep_rotamer = keep_rotamer_tp.view;

  auto keep_block_tp = TPack<int64_t, 2, D>::zeros({n_poses, max_n_blocks});
  auto keep_block = keep_block_tp.view;

  auto rotamer_for_nonmolten_block_tp =
      TPack<int64_t, 2, D>::full({n_poses, max_n_blocks}, -1);
  auto rotamer_for_nonmolten_block = rotamer_for_nonmolten_block_tp.view;

  {  // scope the creation of the first-pass interaction-graph
     // construction so that we can free its memory before
     // re-constructing it in a second pass.

    auto energy1b_tp = TPack<Real, 1, D>::zeros({n_rotamers});
    auto energy1b = energy1b_tp.view;
    auto n_chunks_for_block_tp =
        TPack<int32_t, 2, D>::zeros({n_poses, max_n_blocks});
    auto n_chunks_for_block = n_chunks_for_block_tp.view;

    auto count_n_chunks_for_block = ([=] TMOL_DEVICE_FUNC(int index) {
      int const pose = index / max_n_blocks;
      int const block = index % max_n_blocks;
      int const n_rots = n_rots_for_block[pose][block];
      if (n_rots != 0) {
        int const n_chunks = (n_rots - 1) / chunk_size + 1;
        n_chunks_for_block[pose][block] = n_chunks;
      }
    });
    printf("count_n_chunks_for_block 1\n");
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
      if (block1 == block2) {
        return;
      }
      // Assert: block1 < block2
      respair_is_adjacent[pose][block1][block2] = 1;
    });
    printf("note_adjacent_respairs 1\n");
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

      // We don't have to worry about block1 > block2 as those will not
      // have entries in the sparse_inds input tensors, but we do
      // have to worry about block1 == block2 for one-body energies
      if (respair_is_adjacent[pose][block1][block2]) {
        int const n_chunks1 = n_chunks_for_block[pose][block1];
        int const n_chunks2 = n_chunks_for_block[pose][block2];
        int const n_chunk_pairs = n_chunks1 * n_chunks2;
        n_chunks_for_block_pair[pose][block1][block2] = n_chunk_pairs;
        n_chunks_for_block_pair[pose][block2][block1] = n_chunk_pairs;
      }
    });
    printf("note_n_chunks_for_block_pair 1\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_poses * max_n_blocks * max_n_blocks, note_n_chunks_for_block_pair);

    auto chunk_pair_offset_for_block_pair_tp =
        TPack<int64_t, 3, D>::zeros({n_poses, max_n_blocks, max_n_blocks});
    auto chunk_pair_offset_for_block_pair =
        chunk_pair_offset_for_block_pair_tp.view;

    printf("scan_and_return_total n_chunks_for_block_pair\n");
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
      if (block1 == block2) {
        return;
      }

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

      // multiple threads will write exactly these values to these entries in
      // the chunk_pair_adjacency table
      chunk_pair_adjacency
          [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2] =
              chunk1_size * chunk2_size;
      chunk_pair_adjacency
          [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1] =
              chunk1_size * chunk2_size;
    });
    printf("note_adjacent_chunk_pairs 1\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_sparse_entries, note_adjacent_chunk_pairs);

    auto chunk_pair_offsets_tp =
        TPack<int64_t, 1, D>::zeros({n_adjacent_chunk_pairs_total});
    auto chunk_pair_offsets = chunk_pair_offsets_tp.view;

    printf("scan_and_return_total chunk_pair_adjacency\n");
    int64_t const n_two_body_energies =
        DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
            chunk_pair_adjacency.data(),
            chunk_pair_offsets.data(),
            n_adjacent_chunk_pairs_total,
            mgpu::plus_t<Int>());

    auto energy2b_tp = TPack<Real, 1, D>::zeros({n_two_body_energies});
    auto energy2b = energy2b_tp.view;

    auto record_energies_in_energy1b_and_energy2b =
        ([=] TMOL_DEVICE_FUNC(int index) {
          int const pose = sparse_inds[0][index];
          int const rot1 = sparse_inds[1][index];
          int const rot2 = sparse_inds[2][index];
          Real const energy = sparse_energies[index];
          int const block1 = block_ind_for_rot[rot1];
          int const block2 = block_ind_for_rot[rot2];
          if (block1 == block2) {
            energy1b[rot1] = energy;
          } else {
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
            int const chunk1_size =
                (overhang1 > chunk_size ? chunk_size : overhang1);
            int const chunk2_size =
                (overhang2 > chunk_size ? chunk_size : overhang2);

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

            energy2b
                [chunk_offset_ij + rot_ind_wi_chunk1 * chunk2_size
                 + rot_ind_wi_chunk2] = energy;
            energy2b
                [chunk_offset_ji + rot_ind_wi_chunk2 * chunk1_size
                 + rot_ind_wi_chunk1] = energy;
          }
        });
    printf("record_energies_in_energy1b_and_energy2b 1\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_sparse_entries, record_energies_in_energy1b_and_energy2b);

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
    printf("sentinel_out_non_adjacent_block_pairs 1\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_poses * max_n_blocks * max_n_blocks,
        sentinel_out_non_adjacent_block_pairs);

    auto sentinel_out_non_adjacent_chunk_pairs =
        ([=] TMOL_DEVICE_FUNC(int index) {
          int const n_pairs_for_chunk = chunk_pair_adjacency.data()[index];
          if (n_pairs_for_chunk == 0) {
            chunk_pair_offsets[index] = -1;
          }
        });
    printf("sentinel_out_non_adjacent_chunk_pairs 1\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_adjacent_chunk_pairs_total, sentinel_out_non_adjacent_chunk_pairs);

    // Okay, from here we want to compute the best energy
    // for each rotamer, akin to the Goldstein criterion in DEE:
    // What's the best energy a rotamer can get in any assignment
    // of rotamers.
    // Then we will eliminate any rotamer iff:
    // 1. a rotamer's best energy is worse than +5 kcal/mol, and
    // 2. there is at least one rotamer of that block type with a best energy
    // better +5

    TPack<Real, 1, D> best_energy_for_rot_tp =
        TPack<Real, 1, D>::full({n_rotamers}, 0);
    auto best_energy_for_rot = best_energy_for_rot_tp.view;

    auto compute_best_energy_for_rotamers = ([=] TMOL_DEVICE_FUNC(int index) {
      int const rot1 = index;
      int const block1 = block_ind_for_rot[rot1];
      int const pose = pose_for_rot[rot1];
      int const n_rots_block1 = n_rots_for_block[pose][block1];
      int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
      int const rot_in_block1 = rot1 - rot_offset_for_block[pose][block1];
      int const chunk1 = rot_in_block1 / chunk_size;
      int const rot_ind_wi_chunk1 = rot_in_block1 - chunk1 * chunk_size;
      int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
      int const chunk1_size = (overhang1 > chunk_size ? chunk_size : overhang1);

      Real best_energy = energy1b[rot1];
      for (int block2 = 0; block2 < max_n_blocks; ++block2) {
        if (block2 == block1) {
          continue;
        }
        int64_t const block_pair_chunk_offset =
            chunk_pair_offset_for_block_pair[pose][block1][block2];
        if (block_pair_chunk_offset == -1) {
          continue;
        }
        bool first_rotamer = true;
        Real best_energy_for_rot1_w_block2 = 1234;
        int const n_rots_block2 = n_rots_for_block[pose][block2];
        int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;
        for (int local_rot2 = 0; local_rot2 < n_rots_block2; ++local_rot2) {
          int const chunk2 = local_rot2 / chunk_size;
          int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
          int const chunk2_size =
              (overhang2 > chunk_size ? chunk_size : overhang2);

          int64_t const chunk_pair_offset = chunk_pair_offsets
              [block_pair_chunk_offset + chunk1 * n_chunks2 + chunk2];
          if (chunk_pair_offset == -1) {
            continue;
          }
          Real e2b = energy2b
              [chunk_pair_offset + rot_ind_wi_chunk1 * chunk2_size
               + local_rot2];
          if (first_rotamer || e2b < best_energy_for_rot1_w_block2) {
            best_energy_for_rot1_w_block2 = e2b;
            first_rotamer = false;
          }
        }
        if (!first_rotamer) {
          best_energy += best_energy_for_rot1_w_block2;
        }
      }  // for block2
      best_energy_for_rot[rot1] = best_energy;
    });
    printf("compute_best_energy_for_rotamers\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_rotamers, compute_best_energy_for_rotamers);

    // Now let's figure out the best energy for each block type

    auto best_energy_for_block_type_per_block_tp = TPack<Real, 3, D>::full(
        {n_poses, max_n_blocks, max_n_block_types}, 1234);
    auto best_energy_for_block_type_per_block =
        best_energy_for_block_type_per_block_tp.view;

    // This is just the dumbest way to do this; but it's cheap and it'll work
    auto compute_best_energy_for_block_type_per_block =
        ([=] TMOL_DEVICE_FUNC(int index) {
          int const pose = index / (max_n_blocks * max_n_block_types);
          index = index - pose * max_n_blocks * max_n_block_types;
          int const block = index / max_n_block_types;
          int const block_type = index % max_n_block_types;

          if (n_rots_for_block[pose][block] == 0) {
            return;
          }

          Real best_energy_for_block_type = 1234;
          bool first_rotamer_of_block_type = true;
          int const block_n_rots = n_rots_for_block[pose][block];
          for (int rot_in_block = 0; rot_in_block < block_n_rots;
               ++rot_in_block) {
            int const rot = rot_offset_for_block[pose][block] + rot_in_block;
            if (block_type_ind_for_rot[rot] == block_type) {
              Real energy_for_rot = best_energy_for_rot[rot];
              if (first_rotamer_of_block_type
                  || energy_for_rot < best_energy_for_block_type) {
                best_energy_for_block_type = energy_for_rot;
                first_rotamer_of_block_type = false;
              }
            }
          }
          if (!first_rotamer_of_block_type) {
            best_energy_for_block_type_per_block[pose][block][block_type] =
                best_energy_for_block_type;
          }
        });
    printf("compute_best_energy_for_block_type_per_block\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_poses * max_n_blocks * max_n_block_types,
        compute_best_energy_for_block_type_per_block);

    // Now we ask:
    // for every rotamer, should we keep it?

    auto decide_keep_rotamers = ([=] TMOL_DEVICE_FUNC(int index) {
      int const rot = index;
      int const block = block_ind_for_rot[rot];
      int const pose = pose_for_rot[rot];
      int const block_type = block_type_ind_for_rot[rot];
      Real energy_for_rot = best_energy_for_rot[rot];
      Real best_energy_for_block_type =
          best_energy_for_block_type_per_block[pose][block][block_type];
      if (energy_for_rot < 5.0 || best_energy_for_block_type < 5.0) {
        keep_rotamer[rot] = 1;
      }
    });
    printf("decide_keep_rotamers\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_rotamers, decide_keep_rotamers);

    // Now, last but not least, we will eliminate any blocks
    // which have only a single rotamer, perhaps because it only
    // ever had one rotamer or perhaps because all of its
    // rotamers except one were eliminate by the bump check.

    auto decide_keep_blocks = ([=] TMOL_DEVICE_FUNC(int index) {
      int const pose = index / max_n_blocks;
      int const block = index % max_n_blocks;
      int const block_n_rots = n_rots_for_block[pose][block];
      if (block_n_rots == 0) {
        return;
      }
      int n_kept_rotamers_for_block = 0;
      for (int rot_in_block = 0; rot_in_block < block_n_rots; ++rot_in_block) {
        int const rot = rot_offset_for_block[pose][block] + rot_in_block;
        if (keep_rotamer[rot]) {
          n_kept_rotamers_for_block += 1;
          if (n_kept_rotamers_for_block > 1) {
            break;
          }
        }
      }
      if (n_kept_rotamers_for_block > 1) {
        keep_block[pose][block] = 1;
      } else {
        // we are not keeping this block,
        // so if it has 1 rotamer, then we have to also
        // note that we are not keeping that rotamer
        if (n_kept_rotamers_for_block == 1) {
          for (int rot_in_block = 0; rot_in_block < block_n_rots;
               ++rot_in_block) {
            int const rot = rot_offset_for_block[pose][block] + rot_in_block;
            if (keep_rotamer[rot]) {
              rotamer_for_nonmolten_block[pose][block] = rot;
              keep_rotamer[rot] = 0;
              break;
            }
          }
        }
      }
    });
    printf("decide_keep_blocks\n");
    DeviceDispatch<D>::template forall<launch_t>(
        n_poses * max_n_blocks, decide_keep_blocks);

  }  // end scope of first-pass interaction graph construction
  // This will deallocate the energy1b and energy2b tables

  // OKAY! Now we are ready to rebuild the interaction graph with only the
  // rotamers that we kept.

  // 1st; scan the keep_rotamer table to
  // a) count the total number of rotamers that we are keeping, and
  // b) creating the mapping from old rotamer index to new rotamer index

  auto old_to_new_rotamer_index_tp =
      TPack<int64_t, 1, tmol::Device::CPU>::zeros({n_rotamers});
  auto old_to_new_rotamer_index = old_to_new_rotamer_index_tp.view;
  printf("scan_and_return_total keep_rotamer\n");
  int64_t const n_kept_rotamers =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          keep_rotamer.data(),
          old_to_new_rotamer_index.data(),
          n_rotamers,
          mgpu::plus_t<int64_t>());

  auto new_to_old_rotamer_index_tp =
      TPack<int64_t, 1, D>::zeros({n_kept_rotamers});
  auto new_to_old_rotamer_index = new_to_old_rotamer_index_tp.view;
  auto record_new_rotamer_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const rot = index;
    if (keep_rotamer[rot]) {
      int64_t new_index = old_to_new_rotamer_index[rot];
      new_to_old_rotamer_index[new_index] = rot;
    }
  });
  printf("record_new_rotamer_indices\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_rotamers, record_new_rotamer_indices);

  // Now we are goint to reallocate the energy1b and energy2b tables for the
  // new, smaller set of rotamers, and fill them in with the energies from the
  // old tables

  // now scan the number of molten blocks
  auto block_to_molten_block_inds_tp =
      TPack<Int, 1, D>::zeros({n_poses * max_n_blocks});
  auto block_to_molten_block_inds = block_to_molten_block_inds_tp.view;

  printf("scan_and_return_total\n");
  int const n_molten_blocks =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          keep_block.data(),
          block_to_molten_block_inds.data(),
          n_poses * max_n_blocks,
          mgpu::plus_t<Int>());
  auto molten_block_to_block_inds_tp =
      TPack<Int, 1, D>::zeros({n_molten_blocks});
  auto molten_block_to_block_inds = molten_block_to_block_inds_tp.view;
  auto record_molten_block_indices = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / max_n_blocks;
    int const block = index % max_n_blocks;
    if (keep_block[pose][block]) {
      int molten_block_index = block_to_molten_block_inds[block];
      molten_block_to_block_inds[molten_block_index] =
          block + pose * max_n_blocks;
    }
  });
  printf("record_molten_block_indices\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, record_molten_block_indices);

  // Figure out for each pose what its number of molten blocks is
  auto molten_block_offset_for_pose_tp = TPack<Int, 1, D>::zeros({n_poses});
  auto molten_block_offset_for_pose = molten_block_offset_for_pose_tp.view;
  auto record_n_molten_blocks_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index;
    int last_block_kept = keep_block[pose][max_n_blocks - 1];
    int last_offset_for_pose =
        block_to_molten_block_inds[pose * max_n_blocks + (max_n_blocks - 1)];
    int pose_n_molten_blocks = last_offset_for_pose + last_block_kept;
    molten_block_offset_for_pose[pose + 1] = pose_n_molten_blocks;
  });
  printf("record_n_molten_blocks_per_pose\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses - 1, record_n_molten_blocks_per_pose);
  auto n_molten_blocks_per_pose_tp = TPack<Int, 1, D>::zeros({n_poses});
  auto n_molten_blocks_per_pose = n_molten_blocks_per_pose_tp.view;
  auto compute_n_molten_blocks_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index;
    if (pose < n_poses - 1) {
      n_molten_blocks_per_pose[pose] = molten_block_offset_for_pose[pose + 1]
                                       - molten_block_offset_for_pose[pose];
    } else {
      n_molten_blocks_per_pose[pose] =
          n_molten_blocks - molten_block_offset_for_pose[pose];
    }
  });
  printf("compute_n_molten_blocks_per_pose\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses, compute_n_molten_blocks_per_pose);
  // get maximum number of molten blocks per pose.
  printf("reduce n_molten_blocks_per_pose\n");
  int const max_n_molten_blocks = DeviceDispatch<D>::reduce(
      n_molten_blocks_per_pose.data(), n_poses, mgpu::maximum_t<Int>());

  auto n_bc_rots_per_pose_tp = TPack<Int, 1, D>::zeros({n_poses});
  auto bc_rot_offset_for_pose_tp = TPack<Int, 1, D>::zeros({n_poses});
  auto n_bc_rots_for_molten_block_tp =
      TPack<Int, 2, D>::zeros({n_poses, max_n_molten_blocks});
  auto bc_rot_offset_for_molten_block_tp =
      TPack<Int, 2, D>::zeros({n_poses, max_n_molten_blocks});
  auto molten_block_ind_for_bc_rot_tp =
      TPack<Int, 1, D>::zeros({n_kept_rotamers});

  auto n_bc_rots_per_pose = n_bc_rots_per_pose_tp.view;                  // x
  auto bc_rot_offset_for_pose = bc_rot_offset_for_pose_tp.view;          // x
  auto n_bc_rots_for_molten_block = n_bc_rots_for_molten_block_tp.view;  // x
  auto bc_rot_offset_for_molten_block =
      bc_rot_offset_for_molten_block_tp.view;  // x
  auto molten_block_ind_for_bc_rot = molten_block_ind_for_bc_rot_tp.view;

  // How do we fill the above tensors?
  auto fill_bc_rot_offsets_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index;
    int orig_first_rotamer_for_pose = rot_offset_for_pose[pose];
    int new_first_rotamer_for_pose =
        old_to_new_rotamer_index[orig_first_rotamer_for_pose];
    // This will be the right offset whether or not the first rotamer
    // in this pose has been bump-checked away
    bc_rot_offset_for_pose[pose] = new_first_rotamer_for_pose;
  });
  printf("fill_bc_rot_offsets_per_pose\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses, fill_bc_rot_offsets_per_pose);
  auto fill_n_bc_rots_per_pose = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index;
    int pose_bc_rot_offset = bc_rot_offset_for_pose[pose];
    if (pose < n_poses - 1) {
      n_bc_rots_per_pose[pose] =
          bc_rot_offset_for_pose[pose + 1] - pose_bc_rot_offset;
    } else {
      n_bc_rots_per_pose[pose] = n_kept_rotamers - pose_bc_rot_offset;
    }
  });
  printf("fill_n_bc_rots_per_pose\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses, fill_n_bc_rots_per_pose);
  printf("reduce n_bc_rots_per_pose\n");
  int const max_n_bc_rots_per_pose = DeviceDispatch<D>::reduce(
      n_bc_rots_per_pose.data(), n_poses, mgpu::maximum_t<Int>());
  auto max_n_bump_checked_rotamers_per_pose_tp =
      TPack<int64_t, 1, tmol::Device::CPU>::full({1}, max_n_bc_rots_per_pose);

  auto fill_bc_rot_offset_for_molten_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const molten_block_index = index;
    int const block_and_pose = molten_block_to_block_inds[molten_block_index];
    int const pose = block_and_pose / max_n_blocks;
    int const block = block_and_pose % max_n_blocks;
    int const pose_molten_block_offset = molten_block_offset_for_pose[pose];
    int const block_rot_offset = rot_offset_for_block[pose][block];
    int const bc_rot_for_block = old_to_new_rotamer_index[block_rot_offset];
    int const local_molten_block_index =
        molten_block_index - pose_molten_block_offset;
    bc_rot_offset_for_molten_block[pose][local_molten_block_index] =
        bc_rot_for_block;
  });
  printf("fill_bc_rot_offset_for_molten_block\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_molten_blocks, fill_bc_rot_offset_for_molten_block);
  auto fill_n_bc_rots_for_molten_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const molten_block_index = index;
    int const block_and_pose = molten_block_to_block_inds[molten_block_index];
    int const pose = block_and_pose / max_n_blocks;
    int const block = block_and_pose % max_n_blocks;
    int const local_molten_block_index =
        molten_block_index - molten_block_offset_for_pose[pose];
    int const pose_n_molten_blocks = n_molten_blocks_per_pose[pose];
    int const offset_for_molten_block =
        bc_rot_offset_for_molten_block[pose][local_molten_block_index];
    if (local_molten_block_index < pose_n_molten_blocks) {
      n_bc_rots_for_molten_block[pose][local_molten_block_index] =
          (bc_rot_offset_for_molten_block[pose][local_molten_block_index + 1]
           - offset_for_molten_block);
    } else {
      if (pose < n_poses - 1) {
        n_bc_rots_for_molten_block[pose][local_molten_block_index] =
            (bc_rot_offset_for_pose[pose + 1] - offset_for_molten_block);
      } else {
        n_bc_rots_for_molten_block[pose][local_molten_block_index] =
            (n_kept_rotamers - offset_for_molten_block);
      }
    }
  });
  printf("fill_n_bc_rots_for_molten_block\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_molten_blocks, fill_n_bc_rots_for_molten_block);
  auto fill_molten_block_ind_for_bc_rot = ([=] TMOL_DEVICE_FUNC(int index) {
    int const bc_rot = index;
    int const orig_rot = new_to_old_rotamer_index[bc_rot];
    int const block = block_ind_for_rot[orig_rot];
    int const pose = pose_for_rot[orig_rot];
    // int const molten_block_offset_for_pose =
    // molten_block_offset_for_pose[pose];
    int const molten_block_index =
        block_to_molten_block_inds[pose * max_n_blocks + block];
    molten_block_ind_for_bc_rot[bc_rot] = molten_block_index;
  });
  printf("fill_molten_block_ind_for_bc_rot\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_kept_rotamers, fill_molten_block_ind_for_bc_rot);

  // Now with all of these tensors filled, we are ready to repeat the
  // same process as above to fill in the energy1b and energy2b tables

  auto energy1b_tp = TPack<Real, 1, D>::zeros({n_kept_rotamers});
  auto energy1b = energy1b_tp.view;
  auto n_chunks_for_block_tp =
      TPack<int32_t, 2, D>::zeros({n_poses, max_n_molten_blocks});
  auto n_chunks_for_block = n_chunks_for_block_tp.view;

  auto count_n_chunks_for_block = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / max_n_molten_blocks;
    int const block = index % max_n_molten_blocks;
    int const n_rots = n_bc_rots_for_molten_block[pose][block];
    if (n_rots != 0) {
      int const n_chunks = (n_rots - 1) / chunk_size + 1;
      n_chunks_for_block[pose][block] = n_chunks;
    }
  });
  printf("count_n_chunks_for_block 2\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_blocks, count_n_chunks_for_block);

  auto respair_is_adjacent_tp = TPack<int32_t, 3, D>::zeros(
      {n_poses, max_n_molten_blocks, max_n_molten_blocks});
  auto respair_is_adjacent = respair_is_adjacent_tp.view;

  auto note_adjacent_respairs = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const bc_rot1 = old_to_new_rotamer_index[rot1];
    int const bc_rot2 = old_to_new_rotamer_index[rot2];
    int const kept_rot1 = keep_rotamer[rot1];
    int const kept_rot2 = keep_rotamer[rot2];
    if (!kept_rot1 || !kept_rot2) {
      return;
    }
    int const molten_block1 = molten_block_ind_for_bc_rot[bc_rot1];
    int const molten_block2 = molten_block_ind_for_bc_rot[bc_rot2];
    if (molten_block1 == molten_block2) {
      return;
    }
    // Assert: molten_block1 < molten_block2
    respair_is_adjacent[pose][molten_block1][molten_block2] = 1;
  });
  printf("note_adjacent_respairs 2\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_sparse_entries, note_adjacent_respairs);

  auto n_chunks_for_block_pair_tp = TPack<int64_t, 3, D>::zeros(
      {n_poses, max_n_molten_blocks, max_n_molten_blocks});
  auto n_chunks_for_block_pair = n_chunks_for_block_pair_tp.view;

  auto note_n_chunks_for_block_pair = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = index / (max_n_molten_blocks * max_n_molten_blocks);
    index = index - pose * max_n_molten_blocks * max_n_molten_blocks;
    int const molten_block1 = index / max_n_molten_blocks;
    int const molten_block2 = index % max_n_molten_blocks;

    // We don't have to worry about molten_block1 > molten_block2 as those will
    // not have entries in the sparse_inds input tensors, but we do have to
    // worry about molten_block1 == molten_block2 for one-body energies
    if (respair_is_adjacent[pose][molten_block1][molten_block2]) {
      int const n_chunks1 = n_chunks_for_block[pose][molten_block1];
      int const n_chunks2 = n_chunks_for_block[pose][molten_block2];
      int const n_chunk_pairs = n_chunks1 * n_chunks2;
      n_chunks_for_block_pair[pose][molten_block1][molten_block2] =
          n_chunk_pairs;
      n_chunks_for_block_pair[pose][molten_block2][molten_block1] =
          n_chunk_pairs;
    }
  });
  printf("note_n_chunks_for_block_pair 2\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_molten_blocks * max_n_molten_blocks,
      note_n_chunks_for_block_pair);

  auto chunk_pair_offset_for_block_pair_tp = TPack<int64_t, 3, D>::zeros(
      {n_poses, max_n_molten_blocks, max_n_molten_blocks});
  auto chunk_pair_offset_for_block_pair =
      chunk_pair_offset_for_block_pair_tp.view;

  printf("scan_and_return_total n_chunks_for_block_pair 2\n");
  int const n_adjacent_chunk_pairs_total =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          n_chunks_for_block_pair.data(),
          chunk_pair_offset_for_block_pair.data(),
          n_poses * max_n_molten_blocks * max_n_molten_blocks,
          mgpu::plus_t<Int>());

  auto chunk_pair_adjacency_tp =
      TPack<int64_t, 1, D>::zeros({n_adjacent_chunk_pairs_total});
  auto chunk_pair_adjacency = chunk_pair_adjacency_tp.view;

  auto note_adjacent_chunk_pairs = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const bc_rot1 = old_to_new_rotamer_index[rot1];
    int const bc_rot2 = old_to_new_rotamer_index[rot2];
    int const kept_rot1 = keep_rotamer[rot1];
    int const kept_rot2 = keep_rotamer[rot2];
    if (!kept_rot1 || !kept_rot2) {
      return;
    }

    int const molten_block1 = molten_block_ind_for_bc_rot[bc_rot1];
    int const molten_block2 = molten_block_ind_for_bc_rot[bc_rot2];
    if (molten_block1 == molten_block2) {
      return;
    }

    int const block1_rot_offset =
        bc_rot_offset_for_molten_block[pose][molten_block1];
    int const block2_rot_offset =
        bc_rot_offset_for_molten_block[pose][molten_block2];
    int const local_rot1 = rot1 - block1_rot_offset;
    int const local_rot2 = rot2 - block2_rot_offset;
    int const chunk1 = local_rot1 / chunk_size;
    int const chunk2 = local_rot2 / chunk_size;
    int const n_rots_block1 = n_bc_rots_for_molten_block[pose][molten_block1];
    int const n_rots_block2 = n_bc_rots_for_molten_block[pose][molten_block2];
    int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;

    int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
    int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
    int const chunk1_size = (overhang1 > chunk_size ? chunk_size : overhang1);
    int const chunk2_size = (overhang2 > chunk_size ? chunk_size : overhang2);

    int const block_pair_chunk_offset_ij =
        chunk_pair_offset_for_block_pair[pose][molten_block1][molten_block2];
    int const block_pair_chunk_offset_ji =
        chunk_pair_offset_for_block_pair[pose][molten_block2][molten_block1];

    // multiple threads will write exactly these values to these entries in the
    // chunk_pair_adjacency table

    chunk_pair_adjacency
        [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2] =
            chunk1_size * chunk2_size;

    chunk_pair_adjacency
        [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1] =
            chunk1_size * chunk2_size;
  });
  printf("note_adjacent_chunk_pairs 2\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_sparse_entries, note_adjacent_chunk_pairs);

  auto chunk_pair_offsets_tp =
      TPack<int64_t, 1, D>::zeros({n_adjacent_chunk_pairs_total});
  auto chunk_pair_offsets = chunk_pair_offsets_tp.view;

  printf("scan_and_return_total chunk_pair_adjacency 2\n");
  int64_t const n_two_body_energies =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          chunk_pair_adjacency.data(),
          chunk_pair_offsets.data(),
          n_adjacent_chunk_pairs_total,
          mgpu::plus_t<Int>());

  auto energy2b_tp = TPack<Real, 1, D>::zeros({n_two_body_energies});
  auto energy2b = energy2b_tp.view;

  auto record_energies_in_energy1b_and_energy2b = ([=] TMOL_DEVICE_FUNC(
                                                       int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const bc_rot1 = old_to_new_rotamer_index[rot1];
    int const bc_rot2 = old_to_new_rotamer_index[rot2];
    int const kept_rot1 = keep_rotamer[rot1];
    int const kept_rot2 = keep_rotamer[rot2];
    Real const energy = sparse_energies[index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    int const molten_block1 =
        block_to_molten_block_inds[pose * max_n_blocks + block1];
    int const molten_block2 =
        block_to_molten_block_inds[pose * max_n_blocks + block2];
    int const kept_block1 = keep_block[pose][block1];
    int const kept_block2 = keep_block[pose][block2];

    // Even if both rotamers are the sole remaining rotamer
    // for these two positions that have been relegated to
    // the background, we will not do anything with this energy;
    // we do not keep the background / background energies
    if (!kept_rot1 && !kept_rot2) {
      return;
    }

    if (!kept_block1) {
      // if we did not keep block 1,
      // and if this rotamer is the one
      // we will assign to this position
      // then we should put this two body energy
      // into the one-body energy for the
      // other rotamer.
      if (rot1 == rotamer_for_nonmolten_block[pose][block1]) {
        // then we should put this two body energy into the one-body energy for
        // the other rotamer
        energy1b[bc_rot2] += energy;
      }
      return;
    } else if (!kept_block2) {
      // if we did not keep block 2,
      // and if this rotamer is the one
      // we will assign to this position
      // then we should put this two body energy
      // into the one-body energy for the
      // other rotamer.
      if (rot2 == rotamer_for_nonmolten_block[pose][block2]) {
        // then we should put this two body energy into the one-body energy for
        // the other rotamer
        energy1b[bc_rot1] += energy;
      }
      return;
    }

    if (block1 == block2) {
      energy1b[rot1] = energy;
    } else {
      int const block1_rot_offset =
          bc_rot_offset_for_molten_block[pose][molten_block1];
      int const block2_rot_offset =
          bc_rot_offset_for_molten_block[pose][molten_block2];
      int const local_rot1 = bc_rot1 - block1_rot_offset;
      int const local_rot2 = bc_rot2 - block2_rot_offset;
      int const chunk1 = local_rot1 / chunk_size;
      int const chunk2 = local_rot2 / chunk_size;
      int const n_rots_block1 = n_bc_rots_for_molten_block[pose][molten_block1];
      int const n_rots_block2 = n_bc_rots_for_molten_block[pose][molten_block2];
      int const n_chunks1 = (n_rots_block1 - 1) / chunk_size + 1;
      int const n_chunks2 = (n_rots_block2 - 1) / chunk_size + 1;

      int const overhang1 = n_rots_block1 - chunk1 * chunk_size;
      int const overhang2 = n_rots_block2 - chunk2 * chunk_size;
      int const chunk1_size = (overhang1 > chunk_size ? chunk_size : overhang1);
      int const chunk2_size = (overhang2 > chunk_size ? chunk_size : overhang2);

      int const rot_ind_wi_chunk1 = local_rot1 - chunk1 * chunk_size;
      int const rot_ind_wi_chunk2 = local_rot2 - chunk2 * chunk_size;

      int const block_pair_chunk_offset_ij =
          chunk_pair_offset_for_block_pair[pose][molten_block1][molten_block2];
      int const block_pair_chunk_offset_ji =
          chunk_pair_offset_for_block_pair[pose][molten_block2][molten_block1];

      int const chunk_offset_ij = chunk_pair_offsets
          [block_pair_chunk_offset_ij + chunk1 * n_chunks2 + chunk2];
      int const chunk_offset_ji = chunk_pair_offsets
          [block_pair_chunk_offset_ji + chunk2 * n_chunks1 + chunk1];

      energy2b
          [chunk_offset_ij + rot_ind_wi_chunk1 * chunk2_size
           + rot_ind_wi_chunk2] = energy;
      energy2b
          [chunk_offset_ji + rot_ind_wi_chunk2 * chunk1_size
           + rot_ind_wi_chunk1] = energy;
    }
  });
  printf("record_energies_in_energy1b_and_energy2b 2\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_sparse_entries, record_energies_in_energy1b_and_energy2b);

  // Mark the chunk_pair_offset_for_block_pair that are not adjacent w/ -1s
  // Mark the chunk_pair_offsets that are not adjacent w/ -1s

  auto sentinel_out_non_adjacent_block_pairs =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const pose = index / (max_n_molten_blocks * max_n_molten_blocks);
        index = index - pose * max_n_molten_blocks * max_n_molten_blocks;
        int const block1 = index / max_n_molten_blocks;
        int const block2 = index % max_n_molten_blocks;

        // We don't have to worry about block1 >= block2 as those will not
        // have entries in the sparse_inds input tensors
        if (block1 <= block2 && !respair_is_adjacent[pose][block1][block2]) {
          chunk_pair_offset_for_block_pair[pose][block1][block2] = -1;
          chunk_pair_offset_for_block_pair[pose][block2][block1] = -1;
        }
      });
  printf("sentinel_out_non_adjacent_block_pairs 2\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_poses * max_n_molten_blocks * max_n_molten_blocks,
      sentinel_out_non_adjacent_block_pairs);

  auto sentinel_out_non_adjacent_chunk_pairs =
      ([=] TMOL_DEVICE_FUNC(int index) {
        int const n_pairs_for_chunk = chunk_pair_adjacency.data()[index];
        if (n_pairs_for_chunk == 0) {
          chunk_pair_offsets[index] = -1;
        }
      });
  printf("sentinel_out_non_adjacent_chunk_pairs 2\n");
  DeviceDispatch<D>::template forall<launch_t>(
      n_adjacent_chunk_pairs_total, sentinel_out_non_adjacent_chunk_pairs);

  return std::make_tuple(
      max_n_bump_checked_rotamers_per_pose_tp,
      n_molten_blocks_per_pose_tp,
      n_bc_rots_per_pose_tp,
      bc_rot_offset_for_pose_tp,
      n_bc_rots_for_molten_block_tp,
      bc_rot_offset_for_molten_block_tp,
      molten_block_ind_for_bc_rot_tp,
      rotamer_for_nonmolten_block_tp,
      new_to_old_rotamer_index_tp,
      energy1b_tp,
      chunk_pair_offset_for_block_pair_tp,
      chunk_pair_offsets_tp,
      energy2b_tp);
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
