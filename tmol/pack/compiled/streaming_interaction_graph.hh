#pragma once

#include <tmol/utility/tensor/TensorAccessor.h>
#include <tmol/utility/tensor/TensorPack.h>
#include <tmol/utility/tensor/context_manager.hh>

namespace tmol {
namespace pack {
namespace compiled {

template <
    template <tmol::Device> class DeviceDispatch,
    tmol::Device D,
    typename Real,
    typename Int>
struct StreamingInteractionGraph {
  static auto initialize(
      ContextManager& mgr,
      int chunk_size,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> n_bc_rots_for_molten_block)
      -> std::tuple<
          TPack<int32_t, 3, D>,
          TPack<int64_t, 1, D>,
          TPack<int64_t, 2, D>,
          TPack<int64_t, 2, D>,
          TPack<int64_t, 1, D>,
          TPack<int64_t, 1, D>,
          TPack<int32_t, 1, D>>;

  static void note(
      ContextManager& mgr,
      int chunk_size,
      TView<Int, 2, D> n_rots_for_block,
      TView<Int, 2, D> rot_offset_for_block,
      TView<int32_t, 1, D> block_ind_for_rot,
      TView<int64_t, 2, D> orig_block_to_molten,
      TView<int64_t, 2, D> molten_block_chunk_offset,
      TView<int64_t, 1, D> n_chunks_per_pose,
      TView<int64_t, 1, D> pose_global_chunk_offset,
      TView<int32_t, 3, D> block_adjacency,
      TView<int64_t, 1, D> chunk_pair_keys,
      TView<int32_t, 1, D> hash_overflow,
      TView<int32_t, 2, D> sparse_inds);

  static TPack<int64_t, 1, D> resize_chunk_pair_keys(
      ContextManager& mgr,
      TView<int64_t, 1, D> old_chunk_pair_keys,
      int64_t new_capacity);

  static auto finalize(
      ContextManager& mgr,
      int chunk_size,
      TView<Int, 2, D> n_bc_rots_for_molten_block,
      TView<int64_t, 2, D> molten_block_chunk_offset,
      TView<int64_t, 1, D> n_chunks_per_pose,
      TView<int64_t, 1, D> pose_global_chunk_offset,
      TView<int32_t, 3, D> block_adjacency,
      TView<int64_t, 1, D> chunk_pair_keys)
      -> std::tuple<
          TPack<int64_t, 1, D>,
          TPack<int32_t, 1, D>,
          TPack<int64_t, 1, D>,
          TPack<int64_t, 1, D>,
          TPack<Real, 1, D>>;

  static void accumulate(
      ContextManager& mgr,
      int chunk_size,
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
      TView<Real, 1, D> sparse_energies);
};

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
