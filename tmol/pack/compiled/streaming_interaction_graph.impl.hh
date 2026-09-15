#pragma once

#include <moderngpu/operators.hxx>

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/counting.hh>
#include <tmol/score/common/launch_box_macros.hh>

#include "streaming_interaction_graph.hh"

namespace tmol {
namespace pack {
namespace compiled {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto StreamingInteractionGraph<DeviceDispatch, D, Real, Int>::initialize(
    ContextManager& mgr,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> n_bc_rots_for_molten_block)
    -> std::tuple<TPack<int32_t, 3, D>, TPack<int64_t, 2, D>> {
  int const n_poses = n_rots_for_block.size(0);
  int const max_n_blocks = n_rots_for_block.size(1);
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  int const block_adjacency_words = (max_n_molten_blocks + 31) / 32;

  auto block_adjacency_tp = TPack<int32_t, 3, D>::zeros(
      {n_poses, max_n_molten_blocks, block_adjacency_words});
  auto orig_block_to_molten_tp =
      TPack<int64_t, 2, D>::full({n_poses, max_n_blocks}, -1);
  auto orig_block_to_molten = orig_block_to_molten_tp.view;

  LAUNCH_BOX_32;
  auto initialize_pose = ([=] TMOL_DEVICE_FUNC(int pose) {
    int molten = 0;
    for (int block = 0; block < max_n_blocks; ++block) {
      if (n_rots_for_block[pose][block] <= 1) {
        continue;
      }
      orig_block_to_molten[pose][block] = molten;
      ++molten;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(mgr, n_poses, initialize_pose);

  return {block_adjacency_tp, orig_block_to_molten_tp};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void StreamingInteractionGraph<DeviceDispatch, D, Real, Int>::note(
    ContextManager& mgr,
    TView<int32_t, 1, D> block_ind_for_rot,
    TView<int64_t, 2, D> orig_block_to_molten,
    TView<int32_t, 3, D> block_adjacency,
    TView<int32_t, 2, D> sparse_inds) {
  int64_t const n_entries = sparse_inds.size(1);
  int const n_entries_dispatch = score::common::checked_dispatch_size(
      n_entries, "streaming interaction-graph topology");

  LAUNCH_BOX_32;
  auto note_entry = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    int64_t const molten1 = orig_block_to_molten[pose][block1];
    int64_t const molten2 = orig_block_to_molten[pose][block2];
    if (molten1 < 0 || molten2 < 0 || molten1 == molten2) {
      return;
    }

    int const word2 = molten2 / 32;
    int const word1 = molten1 / 32;
    int32_t const mask2 = int32_t(uint32_t(1) << (molten2 % 32));
    int32_t const mask1 = int32_t(uint32_t(1) << (molten1 % 32));
    DeviceDispatch<D>::bitwise_or(block_adjacency[pose][molten1][word2], mask2);
    DeviceDispatch<D>::bitwise_or(block_adjacency[pose][molten2][word1], mask1);
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, n_entries_dispatch, note_entry);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto StreamingInteractionGraph<DeviceDispatch, D, Real, Int>::finalize(
    ContextManager& mgr,
    int const chunk_size,
    TView<Int, 2, D> n_bc_rots_for_molten_block,
    TView<int32_t, 3, D> block_adjacency)
    -> std::tuple<
        TPack<int64_t, 1, D>,
        TPack<int32_t, 1, D>,
        TPack<int64_t, 1, D>,
        TPack<int64_t, 1, D>,
        TPack<int32_t, 1, D>> {
  int const n_poses = n_bc_rots_for_molten_block.size(0);
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  int const n_rows = score::common::checked_dispatch_product(
      n_poses,
      max_n_molten_blocks,
      "streaming interaction-graph block dispatch");

  LAUNCH_BOX_32;
  auto blocks_are_adjacent =
      ([=] TMOL_DEVICE_FUNC(int pose, int block, int neighbor) {
        int const word = neighbor / 32;
        int32_t const mask = int32_t(uint32_t(1) << (neighbor % 32));
        return (block_adjacency[pose][block][word] & mask) != 0;
      });

  auto n_neighbors_for_block_tp =
      TPack<int64_t, 1, D>::zeros({int64_t(n_rows) + 1});
  auto n_neighbors_for_block = n_neighbors_for_block_tp.view;
  auto count_neighbors = ([=] TMOL_DEVICE_FUNC(int row) {
    int const pose = row / max_n_molten_blocks;
    int const block = row % max_n_molten_blocks;
    int64_t count = 0;
    for (int neighbor = 0; neighbor < max_n_molten_blocks; ++neighbor) {
      count += blocks_are_adjacent(pose, block, neighbor) ? 1 : 0;
    }
    n_neighbors_for_block[row] = count;
  });
  DeviceDispatch<D>::template forall<launch_t>(mgr, n_rows, count_neighbors);

  auto neighbor_row_offsets_tp =
      TPack<int64_t, 1, D>::zeros({int64_t(n_rows) + 1});
  auto neighbor_row_offsets = neighbor_row_offsets_tp.view;
  int64_t const n_directed_edges =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          mgr,
          n_neighbors_for_block.data(),
          neighbor_row_offsets.data(),
          n_rows + 1,
          mgpu::plus_t<int64_t>());
  int const n_directed_edges_dispatch = score::common::checked_dispatch_size(
      n_directed_edges, "streaming interaction-graph edge dispatch");

  auto neighbor_blocks_tp = TPack<int32_t, 1, D>::zeros({n_directed_edges});
  auto n_chunks_for_neighbor_pair_tp =
      TPack<int64_t, 1, D>::zeros({n_directed_edges});
  auto neighbor_blocks = neighbor_blocks_tp.view;
  auto n_chunks_for_neighbor_pair = n_chunks_for_neighbor_pair_tp.view;
  auto fill_neighbors = ([=] TMOL_DEVICE_FUNC(int row) {
    int const pose = row / max_n_molten_blocks;
    int const block = row % max_n_molten_blocks;
    int64_t output = neighbor_row_offsets[row];
    int const n_rots1 = n_bc_rots_for_molten_block[pose][block];
    int const n_chunks1 = n_rots1 == 0 ? 0 : (n_rots1 - 1) / chunk_size + 1;
    for (int neighbor = 0; neighbor < max_n_molten_blocks; ++neighbor) {
      if (!blocks_are_adjacent(pose, block, neighbor)) {
        continue;
      }
      int const n_rots2 = n_bc_rots_for_molten_block[pose][neighbor];
      int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
      neighbor_blocks[output] = neighbor;
      n_chunks_for_neighbor_pair[output] = int64_t(n_chunks1) * n_chunks2;
      ++output;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(mgr, n_rows, fill_neighbors);

  auto neighbor_chunk_offset_offsets_tp =
      TPack<int64_t, 1, D>::zeros({n_directed_edges});
  auto neighbor_chunk_offset_offsets = neighbor_chunk_offset_offsets_tp.view;
  int64_t const n_chunk_pairs =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          mgr,
          n_chunks_for_neighbor_pair.data(),
          neighbor_chunk_offset_offsets.data(),
          n_directed_edges_dispatch,
          mgpu::plus_t<int64_t>());

  auto n_chunk_words_for_neighbor_pair_tp =
      TPack<int64_t, 1, D>::zeros({n_directed_edges});
  auto n_chunk_words_for_neighbor_pair =
      n_chunk_words_for_neighbor_pair_tp.view;
  auto count_chunk_words = ([=] TMOL_DEVICE_FUNC(int edge) {
    n_chunk_words_for_neighbor_pair[edge] =
        (n_chunks_for_neighbor_pair[edge] + 31) / 32;
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_directed_edges_dispatch, count_chunk_words);
  auto neighbor_chunk_bitset_offsets_tp =
      TPack<int64_t, 1, D>::zeros({n_directed_edges});
  auto neighbor_chunk_bitset_offsets = neighbor_chunk_bitset_offsets_tp.view;
  int64_t const n_chunk_words =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          mgr,
          n_chunk_words_for_neighbor_pair.data(),
          neighbor_chunk_bitset_offsets.data(),
          n_directed_edges_dispatch,
          mgpu::plus_t<int64_t>());
  auto chunk_adjacency_tp = TPack<int32_t, 1, D>::zeros({n_chunk_words});

  return {
      neighbor_row_offsets_tp,
      neighbor_blocks_tp,
      neighbor_chunk_offset_offsets_tp,
      neighbor_chunk_bitset_offsets_tp,
      chunk_adjacency_tp};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void StreamingInteractionGraph<DeviceDispatch, D, Real, Int>::note_chunks(
    ContextManager& mgr,
    int const chunk_size,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    TView<int32_t, 1, D> block_ind_for_rot,
    TView<int64_t, 2, D> orig_block_to_molten,
    TView<Int, 2, D> n_bc_rots_for_molten_block,
    TView<int64_t, 1, D> neighbor_row_offsets,
    TView<int32_t, 1, D> neighbor_blocks,
    TView<int64_t, 1, D> neighbor_chunk_bitset_offsets,
    TView<int32_t, 1, D> chunk_adjacency,
    TView<int32_t, 2, D> sparse_inds) {
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  int const n_entries = score::common::checked_dispatch_size(
      sparse_inds.size(1), "streaming interaction-graph chunk topology");

  LAUNCH_BOX_32;
  auto find_neighbor_edge =
      ([=] TMOL_DEVICE_FUNC(int pose, int block, int neighbor) {
        int const row = pose * max_n_molten_blocks + block;
        int64_t lower = neighbor_row_offsets[row];
        int64_t upper = neighbor_row_offsets[row + 1];
        while (lower < upper) {
          int64_t const middle = lower + (upper - lower) / 2;
          if (neighbor_blocks[middle] < neighbor) {
            lower = middle + 1;
          } else {
            upper = middle;
          }
        }
        return lower;
      });
  auto note_entry = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    int64_t const molten1 = orig_block_to_molten[pose][block1];
    int64_t const molten2 = orig_block_to_molten[pose][block2];
    if (molten1 < 0 || molten2 < 0 || molten1 == molten2) {
      return;
    }

    int const chunk1 = (rot1 - rot_offset_for_block[pose][block1]) / chunk_size;
    int const chunk2 = (rot2 - rot_offset_for_block[pose][block2]) / chunk_size;
    int const n_rots1 = n_bc_rots_for_molten_block[pose][molten1];
    int const n_rots2 = n_bc_rots_for_molten_block[pose][molten2];
    int const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
    int64_t const edge12 = find_neighbor_edge(pose, molten1, molten2);
    int64_t const edge21 = find_neighbor_edge(pose, molten2, molten1);
    int64_t const bit12 = int64_t(chunk1) * n_chunks2 + chunk2;
    int64_t const bit21 = int64_t(chunk2) * n_chunks1 + chunk1;
    int32_t const mask12 = int32_t(uint32_t(1) << static_cast<int>(bit12 % 32));
    int32_t const mask21 = int32_t(uint32_t(1) << static_cast<int>(bit21 % 32));
    DeviceDispatch<D>::bitwise_or(
        chunk_adjacency[neighbor_chunk_bitset_offsets[edge12] + bit12 / 32],
        mask12);
    DeviceDispatch<D>::bitwise_or(
        chunk_adjacency[neighbor_chunk_bitset_offsets[edge21] + bit21 / 32],
        mask21);
  });
  DeviceDispatch<D>::template forall_independent<launch_t>(
      mgr, n_entries, note_entry);
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
auto StreamingInteractionGraph<DeviceDispatch, D, Real, Int>::finalize_chunks(
    ContextManager& mgr,
    int const chunk_size,
    TView<Int, 2, D> n_bc_rots_for_molten_block,
    TView<int64_t, 1, D> neighbor_row_offsets,
    TView<int32_t, 1, D> neighbor_blocks,
    TView<int64_t, 1, D> neighbor_chunk_offset_offsets,
    TView<int64_t, 1, D> neighbor_chunk_bitset_offsets,
    TView<int32_t, 1, D> chunk_adjacency)
    -> std::tuple<TPack<int64_t, 1, D>, TPack<Real, 1, D>> {
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  int64_t const n_rows =
      int64_t(n_bc_rots_for_molten_block.size(0)) * max_n_molten_blocks;
  int const n_directed_edges = score::common::checked_dispatch_size(
      neighbor_blocks.size(0), "streaming interaction-graph edge dispatch");

  auto n_chunks_for_neighbor_pair_tp =
      TPack<int64_t, 1, D>::zeros({n_directed_edges});
  auto n_chunks_for_neighbor_pair = n_chunks_for_neighbor_pair_tp.view;
  LAUNCH_BOX_32;
  auto count_chunk_pairs = ([=] TMOL_DEVICE_FUNC(int edge) {
    int64_t lower = 0;
    int64_t upper = n_rows;
    while (lower < upper) {
      int64_t const middle = lower + (upper - lower) / 2;
      if (neighbor_row_offsets[middle + 1] <= edge) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    int const row = lower;
    int const pose = row / max_n_molten_blocks;
    int const block1 = row % max_n_molten_blocks;
    int const block2 = neighbor_blocks[edge];
    int const n_rots1 = n_bc_rots_for_molten_block[pose][block1];
    int const n_rots2 = n_bc_rots_for_molten_block[pose][block2];
    int64_t const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
    int64_t const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
    n_chunks_for_neighbor_pair[edge] = n_chunks1 * n_chunks2;
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_directed_edges, count_chunk_pairs);
  int64_t const n_chunk_pairs =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          mgr,
          n_chunks_for_neighbor_pair.data(),
          neighbor_chunk_offset_offsets.data(),
          n_directed_edges,
          mgpu::plus_t<int64_t>());

  auto chunk_pair_sizes_tp = TPack<int64_t, 1, D>::zeros({n_chunk_pairs});
  auto chunk_pair_sizes = chunk_pair_sizes_tp.view;
  auto fill_chunk_pair_sizes = ([=] TMOL_DEVICE_FUNC(int edge) {
    int64_t lower = 0;
    int64_t upper = n_rows;
    while (lower < upper) {
      int64_t const middle = lower + (upper - lower) / 2;
      if (neighbor_row_offsets[middle + 1] <= edge) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    int const row = lower;
    int const pose = row / max_n_molten_blocks;
    int const block1 = row % max_n_molten_blocks;
    int const block2 = neighbor_blocks[edge];
    int const n_rots1 = n_bc_rots_for_molten_block[pose][block1];
    int const n_rots2 = n_bc_rots_for_molten_block[pose][block2];
    int const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
    int64_t const output_begin = neighbor_chunk_offset_offsets[edge];
    int64_t const bitset_begin = neighbor_chunk_bitset_offsets[edge];
    for (int chunk1 = 0; chunk1 < n_chunks1; ++chunk1) {
      int const chunk1_size = min(chunk_size, n_rots1 - chunk1 * chunk_size);
      for (int chunk2 = 0; chunk2 < n_chunks2; ++chunk2) {
        int64_t const bit = int64_t(chunk1) * n_chunks2 + chunk2;
        int32_t const mask = int32_t(uint32_t(1) << static_cast<int>(bit % 32));
        if ((chunk_adjacency[bitset_begin + bit / 32] & mask) != 0) {
          int const chunk2_size =
              min(chunk_size, n_rots2 - chunk2 * chunk_size);
          chunk_pair_sizes[output_begin + bit] =
              int64_t(chunk1_size) * chunk2_size;
        }
      }
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_directed_edges, fill_chunk_pair_sizes);

  auto chunk_offsets_tp = TPack<int64_t, 1, D>::zeros({n_chunk_pairs});
  auto chunk_offsets = chunk_offsets_tp.view;
  int64_t const n_two_body_energies =
      DeviceDispatch<D>::template scan_and_return_total<mgpu::scan_type_exc>(
          mgr,
          chunk_pair_sizes.data(),
          chunk_offsets.data(),
          n_chunk_pairs,
          mgpu::plus_t<int64_t>());
  auto energy2b_tp = TPack<Real, 1, D>::zeros({n_two_body_energies});
  auto sentinel_empty_chunks = ([=] TMOL_DEVICE_FUNC(int index) {
    if (chunk_pair_sizes[index] == 0) {
      chunk_offsets[index] = -1;
    }
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr,
      score::common::checked_dispatch_size(
          n_chunk_pairs, "streaming interaction-graph chunk dispatch"),
      sentinel_empty_chunks);

  return {chunk_offsets_tp, energy2b_tp};
}

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
void StreamingInteractionGraph<DeviceDispatch, D, Real, Int>::accumulate(
    ContextManager& mgr,
    int const chunk_size,
    TView<Int, 2, D> n_rots_for_block,
    TView<Int, 2, D> rot_offset_for_block,
    TView<int32_t, 1, D> block_ind_for_rot,
    TView<int64_t, 2, D> orig_block_to_molten,
    TView<int64_t, 2, D> rotamer_for_nonmolten_block,
    TView<Int, 2, D> n_bc_rots_for_molten_block,
    TView<Int, 2, D> bc_rot_offset_for_molten_block,
    TView<int64_t, 1, D> neighbor_row_offsets,
    TView<int32_t, 1, D> neighbor_blocks,
    TView<int64_t, 1, D> neighbor_chunk_offset_offsets,
    TView<int64_t, 1, D> chunk_offsets,
    TView<Real, 1, D> bg_bg_energies,
    TView<Real, 1, D> energy1b,
    TView<Real, 1, D> energy2b,
    TView<int32_t, 2, D> sparse_inds,
    TView<Real, 1, D> sparse_energies) {
  int const max_n_molten_blocks = n_bc_rots_for_molten_block.size(1);
  int64_t const n_entries = sparse_inds.size(1);
  int const n_entries_dispatch = score::common::checked_dispatch_size(
      n_entries, "streaming interaction-graph accumulation");

  LAUNCH_BOX_32;
  auto find_neighbor_edge =
      ([=] TMOL_DEVICE_FUNC(int pose, int block, int neighbor) {
        int const row = pose * max_n_molten_blocks + block;
        int64_t lower = neighbor_row_offsets[row];
        int64_t upper = neighbor_row_offsets[row + 1];
        while (lower < upper) {
          int64_t const middle = lower + (upper - lower) / 2;
          if (neighbor_blocks[middle] < neighbor) {
            lower = middle + 1;
          } else {
            upper = middle;
          }
        }
        return lower;
      });
  auto accumulate_entry = ([=] TMOL_DEVICE_FUNC(int index) {
    int const pose = sparse_inds[0][index];
    int const rot1 = sparse_inds[1][index];
    int const rot2 = sparse_inds[2][index];
    int const block1 = block_ind_for_rot[rot1];
    int const block2 = block_ind_for_rot[rot2];
    int64_t const molten1 = orig_block_to_molten[pose][block1];
    int64_t const molten2 = orig_block_to_molten[pose][block2];
    bool const kept_block1 = molten1 >= 0;
    bool const kept_block2 = molten2 >= 0;
    Real const energy = sparse_energies[index];

    if (!kept_block1 && !kept_block2) {
      if (rot1 == rotamer_for_nonmolten_block[pose][block1]
          && rot2 == rotamer_for_nonmolten_block[pose][block2]) {
        score::common::accumulate<D, Real>::add(bg_bg_energies[pose], energy);
      }
      return;
    }

    int64_t bc_rot1 = -1;
    int64_t bc_rot2 = -1;
    if (kept_block1) {
      bc_rot1 = bc_rot_offset_for_molten_block[pose][molten1] + rot1
                - rot_offset_for_block[pose][block1];
    }
    if (kept_block2) {
      bc_rot2 = bc_rot_offset_for_molten_block[pose][molten2] + rot2
                - rot_offset_for_block[pose][block2];
    }
    if (!kept_block1) {
      if (rot1 == rotamer_for_nonmolten_block[pose][block1]) {
        score::common::accumulate<D, Real>::add(energy1b[bc_rot2], energy);
      }
      return;
    }
    if (!kept_block2) {
      if (rot2 == rotamer_for_nonmolten_block[pose][block2]) {
        score::common::accumulate<D, Real>::add(energy1b[bc_rot1], energy);
      }
      return;
    }
    if (block1 == block2) {
      score::common::accumulate<D, Real>::add(energy1b[bc_rot1], energy);
      return;
    }

    int const local_rot1 =
        bc_rot1 - bc_rot_offset_for_molten_block[pose][molten1];
    int const local_rot2 =
        bc_rot2 - bc_rot_offset_for_molten_block[pose][molten2];
    int const chunk1 = local_rot1 / chunk_size;
    int const chunk2 = local_rot2 / chunk_size;
    int const in_chunk1 = local_rot1 - chunk1 * chunk_size;
    int const in_chunk2 = local_rot2 - chunk2 * chunk_size;
    int const n_rots1 = n_bc_rots_for_molten_block[pose][molten1];
    int const n_rots2 = n_bc_rots_for_molten_block[pose][molten2];
    int const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
    int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
    int const chunk1_size = min(chunk_size, n_rots1 - chunk1 * chunk_size);
    int const chunk2_size = min(chunk_size, n_rots2 - chunk2 * chunk_size);
    int64_t const edge12 = find_neighbor_edge(pose, molten1, molten2);
    int64_t const edge21 = find_neighbor_edge(pose, molten2, molten1);
    int64_t const offset12 = chunk_offsets
        [neighbor_chunk_offset_offsets[edge12] + chunk1 * n_chunks2 + chunk2];
    int64_t const offset21 = chunk_offsets
        [neighbor_chunk_offset_offsets[edge21] + chunk2 * n_chunks1 + chunk1];
    score::common::accumulate<D, Real>::add(
        energy2b[offset12 + in_chunk1 * chunk2_size + in_chunk2], energy);
    score::common::accumulate<D, Real>::add(
        energy2b[offset21 + in_chunk2 * chunk1_size + in_chunk1], energy);
  });
  DeviceDispatch<D>::template forall<launch_t>(
      mgr, n_entries_dispatch, accumulate_entry);
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
