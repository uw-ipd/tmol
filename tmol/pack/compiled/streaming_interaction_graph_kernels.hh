#pragma once

// Per-entry bodies of the streaming interaction-graph builder.
//
// Each pass of the builder walks a table of rotamer-pair energies, one entry
// per thread: noting which blocks interact, noting which rotamer chunks
// inside those block pairs do, and accumulating the energies into the
// one-body and chunked two-body tables. The bodies live here, against an
// explicit context, so the CPU and CUDA lambdas in
// streaming_interaction_graph.impl.hh and the Metal entry points in
// streaming_interaction_graph_mps_kernels.metal.in run the same source.
//
// Templated on the view so the field list is declared exactly once: the host
// instantiates the context with TView, and a Metal translation unit -- which
// cannot include the torch tensor layer -- with tmol::portable::MetalView.

#include <tmol/score/common/accumulate.hh>
#include <tmol/score/common/diamond_macros.hh>
#include <tmol/utility/portable/device.hh>
#include <tmol/utility/portable/metal_view.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <template <typename, int> class View, typename Real, typename Int>
struct InteractionGraphEntryCtxT {
  // The table being walked: [3, n] (pose, rot1, rot2) and [n] energies
  View<int32_t, 2> sparse_inds;
  View<Real, 1> sparse_energies;

  // Rotamer and block numbering
  View<int32_t, 1> block_ind_for_rot;
  View<int64_t, 2> orig_block_to_molten;
  View<Int, 2> rot_offset_for_block;
  View<int64_t, 2> rotamer_for_nonmolten_block;
  View<Int, 2> n_bc_rots_for_molten_block;
  View<Int, 2> bc_rot_offset_for_molten_block;

  // Graph topology: block-neighbor CSR and per-edge chunk layout
  View<int64_t, 1> neighbor_row_offsets;
  View<int32_t, 1> neighbor_blocks;
  View<int64_t, 1> neighbor_chunk_offset_offsets;
  View<int64_t, 1> neighbor_chunk_bitset_offsets;
  View<int64_t, 1> chunk_offsets;

  // Outputs: adjacency bitsets, then the energy tables
  View<int32_t, 3> block_adjacency;
  View<int32_t, 1> chunk_adjacency;
  View<Real, 1> bg_bg_energies;
  View<Real, 1> energy1b;
  View<Real, 1> energy2b;

  int chunk_size;
  int max_n_molten_blocks;

  template <typename F>
  TMOL_HD void visit_views(F f) {
    f(sparse_inds);
    f(sparse_energies);
    f(block_ind_for_rot);
    f(orig_block_to_molten);
    f(rot_offset_for_block);
    f(rotamer_for_nonmolten_block);
    f(n_bc_rots_for_molten_block);
    f(bc_rot_offset_for_molten_block);
    f(neighbor_row_offsets);
    f(neighbor_blocks);
    f(neighbor_chunk_offset_offsets);
    f(neighbor_chunk_bitset_offsets);
    f(chunk_offsets);
    f(block_adjacency);
    f(chunk_adjacency);
    f(bg_bg_energies);
    f(energy1b);
    f(energy2b);
  }

  enum { n_views = 18 };
  enum { scalar_chunk_size = 0, scalar_max_n_molten_blocks = 1, n_scalars = 2 };
};

// The CSR position of the edge block -> neighbor; the edge must exist.
template <typename Ctx>
TMOL_HD int64_t ig_find_neighbor_edge(
    TMOL_THREAD Ctx const& ctx, int pose, int block, int neighbor) {
  int const row = pose * ctx.max_n_molten_blocks + block;
  int64_t lower = ctx.neighbor_row_offsets[row];
  int64_t upper = ctx.neighbor_row_offsets[row + 1];
  while (lower < upper) {
    int64_t const middle = lower + (upper - lower) / 2;
    if (ctx.neighbor_blocks[middle] < neighbor) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  return lower;
}

// Pass one: mark the two molten blocks of this entry adjacent.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Ctx>
TMOL_HD void ig_note_block_pair(TMOL_THREAD Ctx const& ctx, int index) {
  int const pose = ctx.sparse_inds[0][index];
  int const rot1 = ctx.sparse_inds[1][index];
  int const rot2 = ctx.sparse_inds[2][index];
  int const block1 = ctx.block_ind_for_rot[rot1];
  int const block2 = ctx.block_ind_for_rot[rot2];
  int64_t const molten1 = ctx.orig_block_to_molten[pose][block1];
  int64_t const molten2 = ctx.orig_block_to_molten[pose][block2];
  if (molten1 < 0 || molten2 < 0 || molten1 == molten2) {
    return;
  }
  int const word2 = int(molten2 / 32);
  int const word1 = int(molten1 / 32);
  int32_t const mask2 = int32_t(uint32_t(1) << int(molten2 % 32));
  int32_t const mask1 = int32_t(uint32_t(1) << int(molten1 % 32));
  DeviceDispatch<D>::bitwise_or(
      ctx.block_adjacency[pose][molten1][word2], mask2);
  DeviceDispatch<D>::bitwise_or(
      ctx.block_adjacency[pose][molten2][word1], mask1);
}

// Pass two: mark the rotamer-chunk pair of this entry occupied, in both
// directions of its block edge.
template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Ctx>
TMOL_HD void ig_note_chunk_pair(TMOL_THREAD Ctx const& ctx, int index) {
  int const pose = ctx.sparse_inds[0][index];
  int const rot1 = ctx.sparse_inds[1][index];
  int const rot2 = ctx.sparse_inds[2][index];
  int const block1 = ctx.block_ind_for_rot[rot1];
  int const block2 = ctx.block_ind_for_rot[rot2];
  int64_t const molten1 = ctx.orig_block_to_molten[pose][block1];
  int64_t const molten2 = ctx.orig_block_to_molten[pose][block2];
  if (molten1 < 0 || molten2 < 0 || molten1 == molten2) {
    return;
  }
  int const chunk_size = ctx.chunk_size;
  int const chunk1 =
      int(rot1 - ctx.rot_offset_for_block[pose][block1]) / chunk_size;
  int const chunk2 =
      int(rot2 - ctx.rot_offset_for_block[pose][block2]) / chunk_size;
  int const n_rots1 = int(ctx.n_bc_rots_for_molten_block[pose][molten1]);
  int const n_rots2 = int(ctx.n_bc_rots_for_molten_block[pose][molten2]);
  int const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
  int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
  int64_t const edge12 =
      ig_find_neighbor_edge(ctx, pose, int(molten1), int(molten2));
  int64_t const edge21 =
      ig_find_neighbor_edge(ctx, pose, int(molten2), int(molten1));
  int64_t const bit12 = int64_t(chunk1) * n_chunks2 + chunk2;
  int64_t const bit21 = int64_t(chunk2) * n_chunks1 + chunk1;
  int32_t const mask12 = int32_t(uint32_t(1) << int(bit12 % 32));
  int32_t const mask21 = int32_t(uint32_t(1) << int(bit21 % 32));
  DeviceDispatch<D>::bitwise_or(
      ctx.chunk_adjacency
          [ctx.neighbor_chunk_bitset_offsets[edge12] + bit12 / 32],
      mask12);
  DeviceDispatch<D>::bitwise_or(
      ctx.chunk_adjacency
          [ctx.neighbor_chunk_bitset_offsets[edge21] + bit21 / 32],
      mask21);
}

// Pass three: add this entry's energy where it belongs -- the background
// constant, a one-body energy, or both directions of a two-body energy.
template <tmol::Device D, typename Real, typename Ctx>
TMOL_HD void ig_accumulate_entry(TMOL_THREAD Ctx const& ctx, int index) {
  using score::common::accumulate;
  int const pose = ctx.sparse_inds[0][index];
  int const rot1 = ctx.sparse_inds[1][index];
  int const rot2 = ctx.sparse_inds[2][index];
  int const block1 = ctx.block_ind_for_rot[rot1];
  int const block2 = ctx.block_ind_for_rot[rot2];
  int64_t const molten1 = ctx.orig_block_to_molten[pose][block1];
  int64_t const molten2 = ctx.orig_block_to_molten[pose][block2];
  bool const kept_block1 = molten1 >= 0;
  bool const kept_block2 = molten2 >= 0;
  Real const energy = ctx.sparse_energies[index];

  if (!kept_block1 && !kept_block2) {
    if (rot1 == ctx.rotamer_for_nonmolten_block[pose][block1]
        && rot2 == ctx.rotamer_for_nonmolten_block[pose][block2]) {
      accumulate<D, Real>::add(ctx.bg_bg_energies[pose], energy);
    }
    return;
  }

  int64_t bc_rot1 = -1;
  int64_t bc_rot2 = -1;
  if (kept_block1) {
    bc_rot1 = ctx.bc_rot_offset_for_molten_block[pose][molten1] + rot1
              - ctx.rot_offset_for_block[pose][block1];
  }
  if (kept_block2) {
    bc_rot2 = ctx.bc_rot_offset_for_molten_block[pose][molten2] + rot2
              - ctx.rot_offset_for_block[pose][block2];
  }
  if (!kept_block1) {
    if (rot1 == ctx.rotamer_for_nonmolten_block[pose][block1]) {
      accumulate<D, Real>::add(ctx.energy1b[bc_rot2], energy);
    }
    return;
  }
  if (!kept_block2) {
    if (rot2 == ctx.rotamer_for_nonmolten_block[pose][block2]) {
      accumulate<D, Real>::add(ctx.energy1b[bc_rot1], energy);
    }
    return;
  }
  if (block1 == block2) {
    accumulate<D, Real>::add(ctx.energy1b[bc_rot1], energy);
    return;
  }

  int const chunk_size = ctx.chunk_size;
  int const local_rot1 =
      int(bc_rot1 - ctx.bc_rot_offset_for_molten_block[pose][molten1]);
  int const local_rot2 =
      int(bc_rot2 - ctx.bc_rot_offset_for_molten_block[pose][molten2]);
  int const chunk1 = local_rot1 / chunk_size;
  int const chunk2 = local_rot2 / chunk_size;
  int const in_chunk1 = local_rot1 - chunk1 * chunk_size;
  int const in_chunk2 = local_rot2 - chunk2 * chunk_size;
  int const n_rots1 = int(ctx.n_bc_rots_for_molten_block[pose][molten1]);
  int const n_rots2 = int(ctx.n_bc_rots_for_molten_block[pose][molten2]);
  int const n_chunks1 = (n_rots1 - 1) / chunk_size + 1;
  int const n_chunks2 = (n_rots2 - 1) / chunk_size + 1;
  int const chunk1_size = min(chunk_size, n_rots1 - chunk1 * chunk_size);
  int const chunk2_size = min(chunk_size, n_rots2 - chunk2 * chunk_size);
  int64_t const edge12 =
      ig_find_neighbor_edge(ctx, pose, int(molten1), int(molten2));
  int64_t const edge21 =
      ig_find_neighbor_edge(ctx, pose, int(molten2), int(molten1));
  int64_t const offset12 = ctx.chunk_offsets
                               [ctx.neighbor_chunk_offset_offsets[edge12]
                                + chunk1 * n_chunks2 + chunk2];
  int64_t const offset21 = ctx.chunk_offsets
                               [ctx.neighbor_chunk_offset_offsets[edge21]
                                + chunk2 * n_chunks1 + chunk1];
  accumulate<D, Real>::add(
      ctx.energy2b[offset12 + in_chunk1 * chunk2_size + in_chunk2], energy);
  accumulate<D, Real>::add(
      ctx.energy2b[offset21 + in_chunk2 * chunk1_size + in_chunk1], energy);
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
