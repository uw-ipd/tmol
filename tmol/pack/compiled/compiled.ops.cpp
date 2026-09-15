#include <torch/script.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/function.h>        // ??
#include <torch/csrc/autograd/saved_variable.h>  // ??
#include <torch/types.h>

#include <tmol/utility/nvtx.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/tensor/context_manager.hh>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/device_operations.hh>

#include "annealer.hh"
#include "simulated_annealing.hh"
#include "streaming_interaction_graph.hh"

namespace tmol {
namespace pack {
namespace compiled {

ContextManager mgr;
using torch::Tensor;

std::vector<Tensor> build_interaction_graph(
    int64_t const bump_check,
    int64_t const chunk_size,
    int64_t const max_n_block_types,
    Tensor n_rots_for_pose,
    Tensor rot_offset_for_pose,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    Tensor pose_for_rot,
    Tensor block_type_ind_for_rot,
    Tensor block_ind_for_rot,
    Tensor sparse_inds,
    Tensor sparse_energies,
    int64_t const verbose) {
  nvtx_range_push("pack_build_ig");

  at::Tensor max_n_bump_checked_rotamers_per_pose;
  at::Tensor n_molten_blocks_per_pose;
  at::Tensor n_bc_rots_per_pose;
  at::Tensor bc_rot_offset_for_pose;
  at::Tensor n_bc_rots_for_molten_block;
  at::Tensor bc_rot_offset_for_molten_block;
  at::Tensor molten_block_ind_for_bc_rot;
  at::Tensor rotamer_for_nonmolten_block;
  at::Tensor bc_rot_to_orig_rot;

  at::Tensor bg_bg_energies;
  at::Tensor energy1b;
  at::Tensor neighbor_row_offsets;
  at::Tensor neighbor_blocks;
  at::Tensor neighbor_chunk_offset_offsets;
  at::Tensor chunk_pair_offset;
  at::Tensor energy2b;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      sparse_energies.options(), "pack_build_ig", ([&] {
        constexpr tmol::Device Dev = device_t;
        using Real = scalar_t;

        auto result = InteractionGraphBuilder<
            score::common::DeviceOperations,
            Dev,
            Real,
            Int>::

            f(mgr,
              bump_check,
              chunk_size,
              max_n_block_types,
              TCAST(n_rots_for_pose),
              TCAST(rot_offset_for_pose),
              TCAST(n_rots_for_block),
              TCAST(rot_offset_for_block),
              TCAST(pose_for_rot),
              TCAST(block_type_ind_for_rot),
              TCAST(block_ind_for_rot),
              TCAST(sparse_inds),
              TCAST(sparse_energies),
              verbose);

        max_n_bump_checked_rotamers_per_pose = std::get<0>(result).tensor;
        n_molten_blocks_per_pose = std::get<1>(result).tensor;
        n_bc_rots_per_pose = std::get<2>(result).tensor;
        bc_rot_offset_for_pose = std::get<3>(result).tensor;
        n_bc_rots_for_molten_block = std::get<4>(result).tensor;
        bc_rot_offset_for_molten_block = std::get<5>(result).tensor;
        molten_block_ind_for_bc_rot = std::get<6>(result).tensor;
        rotamer_for_nonmolten_block = std::get<7>(result).tensor;
        bc_rot_to_orig_rot = std::get<8>(result).tensor;
        bg_bg_energies = std::get<9>(result).tensor;
        energy1b = std::get<10>(result).tensor;
        neighbor_row_offsets = std::get<11>(result).tensor;
        neighbor_blocks = std::get<12>(result).tensor;
        neighbor_chunk_offset_offsets = std::get<13>(result).tensor;
        chunk_pair_offset = std::get<14>(result).tensor;
        energy2b = std::get<15>(result).tensor;
      }));
  std::vector<torch::Tensor> result(
      {max_n_bump_checked_rotamers_per_pose,
       n_molten_blocks_per_pose,
       n_bc_rots_per_pose,
       bc_rot_offset_for_pose,
       n_bc_rots_for_molten_block,
       bc_rot_offset_for_molten_block,
       molten_block_ind_for_bc_rot,
       rotamer_for_nonmolten_block,
       bc_rot_to_orig_rot,
       bg_bg_energies,
       energy1b,
       neighbor_row_offsets,
       neighbor_blocks,
       neighbor_chunk_offset_offsets,
       chunk_pair_offset,
       energy2b});
  return result;
}

std::vector<Tensor> initialize_interaction_graph_topology(
    Tensor n_rots_for_block,
    Tensor n_bc_rots_for_molten_block,
    Tensor energy_template) {
  std::vector<Tensor> result;
  TMOL_DISPATCH_FLOATING_DEVICE(
      energy_template.options(), "pack_initialize_ig_topology", ([&] {
        constexpr tmol::Device Dev = device_t;
        auto topology = StreamingInteractionGraph<
            score::common::DeviceOperations,
            Dev,
            scalar_t,
            int64_t>::
            initialize(
                mgr,
                TCAST(n_rots_for_block),
                TCAST(n_bc_rots_for_molten_block));
        result = {std::get<0>(topology).tensor, std::get<1>(topology).tensor};
      }));
  return result;
}

std::vector<Tensor> note_interaction_graph_topology(
    Tensor block_ind_for_rot,
    Tensor orig_block_to_molten,
    Tensor block_adjacency,
    Tensor sparse_inds,
    Tensor energy_template) {
  TMOL_DISPATCH_FLOATING_DEVICE(
      energy_template.options(), "pack_note_ig_topology", ([&] {
        constexpr tmol::Device Dev = device_t;
        StreamingInteractionGraph<
            score::common::DeviceOperations,
            Dev,
            scalar_t,
            int64_t>::
            note(
                mgr,
                TCAST(block_ind_for_rot),
                TCAST(orig_block_to_molten),
                TCAST(block_adjacency),
                TCAST(sparse_inds));
      }));
  return {block_adjacency};
}

std::vector<Tensor> finalize_interaction_graph_topology(
    int64_t const chunk_size,
    Tensor n_bc_rots_for_molten_block,
    Tensor block_adjacency,
    Tensor energy_template) {
  std::vector<Tensor> result;
  TMOL_DISPATCH_FLOATING_DEVICE(
      energy_template.options(), "pack_finalize_ig_topology", ([&] {
        constexpr tmol::Device Dev = device_t;
        auto topology = StreamingInteractionGraph<
            score::common::DeviceOperations,
            Dev,
            scalar_t,
            int64_t>::
            finalize(
                mgr,
                chunk_size,
                TCAST(n_bc_rots_for_molten_block),
                TCAST(block_adjacency));
        result = {
            std::get<0>(topology).tensor,
            std::get<1>(topology).tensor,
            std::get<2>(topology).tensor,
            std::get<3>(topology).tensor,
            std::get<4>(topology).tensor};
      }));
  return result;
}

std::vector<Tensor> note_interaction_graph_chunk_topology(
    int64_t const chunk_size,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    Tensor block_ind_for_rot,
    Tensor orig_block_to_molten,
    Tensor n_bc_rots_for_molten_block,
    Tensor neighbor_row_offsets,
    Tensor neighbor_blocks,
    Tensor neighbor_chunk_bitset_offsets,
    Tensor chunk_adjacency,
    Tensor sparse_inds,
    Tensor energy_template) {
  TMOL_DISPATCH_FLOATING_DEVICE(
      energy_template.options(), "pack_note_ig_chunk_topology", ([&] {
        constexpr tmol::Device Dev = device_t;
        StreamingInteractionGraph<
            score::common::DeviceOperations,
            Dev,
            scalar_t,
            int64_t>::
            note_chunks(
                mgr,
                chunk_size,
                TCAST(n_rots_for_block),
                TCAST(rot_offset_for_block),
                TCAST(block_ind_for_rot),
                TCAST(orig_block_to_molten),
                TCAST(n_bc_rots_for_molten_block),
                TCAST(neighbor_row_offsets),
                TCAST(neighbor_blocks),
                TCAST(neighbor_chunk_bitset_offsets),
                TCAST(chunk_adjacency),
                TCAST(sparse_inds));
      }));
  return {chunk_adjacency};
}

std::vector<Tensor> finalize_interaction_graph_chunk_topology(
    int64_t const chunk_size,
    Tensor n_bc_rots_for_molten_block,
    Tensor neighbor_row_offsets,
    Tensor neighbor_blocks,
    Tensor neighbor_chunk_offset_offsets,
    Tensor neighbor_chunk_bitset_offsets,
    Tensor chunk_adjacency,
    Tensor energy_template) {
  std::vector<Tensor> result;
  TMOL_DISPATCH_FLOATING_DEVICE(
      energy_template.options(), "pack_finalize_ig_chunk_topology", ([&] {
        constexpr tmol::Device Dev = device_t;
        auto topology = StreamingInteractionGraph<
            score::common::DeviceOperations,
            Dev,
            scalar_t,
            int64_t>::
            finalize_chunks(
                mgr,
                chunk_size,
                TCAST(n_bc_rots_for_molten_block),
                TCAST(neighbor_row_offsets),
                TCAST(neighbor_blocks),
                TCAST(neighbor_chunk_offset_offsets),
                TCAST(neighbor_chunk_bitset_offsets),
                TCAST(chunk_adjacency));
        result = {std::get<0>(topology).tensor, std::get<1>(topology).tensor};
      }));
  return result;
}

std::vector<Tensor> accumulate_interaction_graph_entries(
    int64_t const chunk_size,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    Tensor block_ind_for_rot,
    Tensor orig_block_to_molten,
    Tensor rotamer_for_nonmolten_block,
    Tensor n_bc_rots_for_molten_block,
    Tensor bc_rot_offset_for_molten_block,
    Tensor neighbor_row_offsets,
    Tensor neighbor_blocks,
    Tensor neighbor_chunk_offset_offsets,
    Tensor chunk_offsets,
    Tensor bg_bg_energies,
    Tensor energy1b,
    Tensor energy2b,
    Tensor sparse_inds,
    Tensor sparse_energies) {
  TMOL_DISPATCH_FLOATING_DEVICE(
      sparse_energies.options(), "pack_accumulate_ig_entries", ([&] {
        constexpr tmol::Device Dev = device_t;
        StreamingInteractionGraph<
            score::common::DeviceOperations,
            Dev,
            scalar_t,
            int64_t>::
            accumulate(
                mgr,
                chunk_size,
                TCAST(n_rots_for_block),
                TCAST(rot_offset_for_block),
                TCAST(block_ind_for_rot),
                TCAST(orig_block_to_molten),
                TCAST(rotamer_for_nonmolten_block),
                TCAST(n_bc_rots_for_molten_block),
                TCAST(bc_rot_offset_for_molten_block),
                TCAST(neighbor_row_offsets),
                TCAST(neighbor_blocks),
                TCAST(neighbor_chunk_offset_offsets),
                TCAST(chunk_offsets),
                TCAST(bg_bg_energies),
                TCAST(energy1b),
                TCAST(energy2b),
                TCAST(sparse_inds),
                TCAST(sparse_energies));
      }));
  return {bg_bg_energies, energy1b, energy2b};
}

std::vector<Tensor> anneal(
    int64_t max_n_rotamers_per_pose,
    Tensor pose_n_res,
    Tensor pose_n_rotamers,
    Tensor pose_rotamer_offset,
    Tensor n_rotamers_for_res,
    Tensor oneb_offsets,
    Tensor res_for_rot,
    int64_t chunk_size,
    Tensor neighbor_row_offsets,
    Tensor neighbor_blocks,
    Tensor neighbor_chunk_offset_offsets,
    Tensor chunk_offsets,
    Tensor energy1b,
    Tensor energy2b) {
  nvtx_range_push("pack_anneal");
  at::Tensor scores;
  at::Tensor rotamer_assignments;

  TMOL_DISPATCH_FLOATING_DEVICE(energy1b.options(), "pack_anneal", ([&] {
                                  constexpr tmol::Device Dev = device_t;

                                  auto result = AnnealerDispatch<Dev>::forward(
                                      mgr,
                                      max_n_rotamers_per_pose,
                                      TCAST(pose_n_res),
                                      TCAST(pose_n_rotamers),
                                      TCAST(pose_rotamer_offset),
                                      TCAST(n_rotamers_for_res),
                                      TCAST(oneb_offsets),
                                      TCAST(res_for_rot),
                                      chunk_size,
                                      TCAST(neighbor_row_offsets),
                                      TCAST(neighbor_blocks),
                                      TCAST(neighbor_chunk_offset_offsets),
                                      TCAST(chunk_offsets),
                                      TCAST(energy1b),
                                      TCAST(energy2b));
                                  scores = std::get<0>(result).tensor;
                                  rotamer_assignments =
                                      std::get<1>(result).tensor;
                                }));

  std::vector<torch::Tensor> result({scores, rotamer_assignments});
  return result;
}

TPack<float, 2, tmol::Device::CPU> compute_energies_for_assignments(
    TView<int, 2, tmol::Device::CPU> n_rotamers_for_res,
    TView<int, 2, tmol::Device::CPU> oneb_offsets,
    int32_t chunk_size,
    TView<int64_t, 1, tmol::Device::CPU> neighbor_row_offsets,
    TView<int32_t, 1, tmol::Device::CPU> neighbor_blocks,
    TView<int64_t, 1, tmol::Device::CPU> neighbor_chunk_offset_offsets,
    TView<int64_t, 1, tmol::Device::CPU> chunk_offsets,
    TView<float, 1, tmol::Device::CPU> energy1b,
    TView<float, 1, tmol::Device::CPU> energy2b,
    TView<int, 3, tmol::Device::CPU> rotamer_assignments) {
  int n_poses = rotamer_assignments.size(0);
  int n_traj = rotamer_assignments.size(1);
  auto scores_t = TPack<float, 2, tmol::Device::CPU>::zeros({n_poses, n_traj});
  auto scores = scores_t.view;
  for (int pose = 0; pose < n_poses; ++pose) {
    for (int i = 0; i < n_traj; ++i) {
      scores[pose][i] = total_energy_for_assignment(
          n_rotamers_for_res[pose],
          oneb_offsets[pose],
          chunk_size,
          pose,
          n_rotamers_for_res.size(1),
          neighbor_row_offsets,
          neighbor_blocks,
          neighbor_chunk_offset_offsets,
          chunk_offsets,
          energy1b,
          energy2b,
          rotamer_assignments[pose][i]);
    }
  }
  return scores_t;
}

torch::Tensor validate_energies(
    Tensor nrotamers_for_res,
    Tensor oneb_offsets,
    int64_t chunk_size,
    Tensor neighbor_row_offsets,
    Tensor neighbor_blocks,
    Tensor neighbor_chunk_offset_offsets,
    Tensor chunk_offsets,
    Tensor energy1b,
    Tensor energy2b,
    Tensor rotamer_assignments) {
  auto result = compute_energies_for_assignments(
      TCAST(nrotamers_for_res),
      TCAST(oneb_offsets),
      int32_t(chunk_size),
      TCAST(neighbor_row_offsets),
      TCAST(neighbor_blocks),
      TCAST(neighbor_chunk_offset_offsets),
      TCAST(chunk_offsets),
      TCAST(energy1b),
      TCAST(energy2b),
      TCAST(rotamer_assignments));
  return result.tensor;
}

// See https://stackoverflow.com/a/3221914
TORCH_LIBRARY(tmol_pack, m) {
  m.def("pack_anneal", &anneal);
  m.def("validate_energies", &validate_energies);
  m.def("build_interaction_graph", &build_interaction_graph);
  m.def(
      "initialize_interaction_graph_topology",
      &initialize_interaction_graph_topology);
  m.def("note_interaction_graph_topology", &note_interaction_graph_topology);
  m.def(
      "finalize_interaction_graph_topology",
      &finalize_interaction_graph_topology);
  m.def(
      "note_interaction_graph_chunk_topology",
      &note_interaction_graph_chunk_topology);
  m.def(
      "finalize_interaction_graph_chunk_topology",
      &finalize_interaction_graph_chunk_topology);
  m.def(
      "accumulate_interaction_graph_entries",
      &accumulate_interaction_graph_entries);
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
