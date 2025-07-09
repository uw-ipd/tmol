#include <torch/script.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/function.h>        // ??
#include <torch/csrc/autograd/saved_variable.h>  // ??
#include <torch/types.h>

#include <tmol/utility/nvtx.hh>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>

#include "annealer.hh"
#include "simulated_annealing.hh"

namespace tmol {
namespace pack {
namespace compiled {

using torch::Tensor;

std::vector<Tensor> build_interaction_graph(
    int64_t const chunk_size,
    Tensor n_rots_for_pose,
    Tensor rot_offset_for_pose,
    Tensor n_rots_for_block,
    Tensor rot_offset_for_block,
    Tensor pose_for_rot,
    Tensor block_type_ind_for_rot,
    Tensor block_ind_for_rot,
    Tensor sparse_inds,
    Tensor sparse_energies) {
  nvtx_range_push("pack_build_ig");
  at::Tensor chunk_pair_offset_for_block_pair;
  at::Tensor chunk_pair_offset;
  at::Tensor energy2b;

  using Int = int64_t;

  TMOL_DISPATCH_FLOATING_DEVICE(
      sparse_energies.options(), "pack_build_ig", ([&] {
        constexpr tmol::Device Dev = device_t;
        using Real = scalar_t;

        std::cout << "Hello!" << std::endl;
        auto result = InteractionGraphBuilder<
            score::common::DeviceOperations,
            Dev,
            Real,
            Int>::
            f(chunk_size,
              TCAST(n_rots_for_pose),
              TCAST(rot_offset_for_pose),
              TCAST(n_rots_for_block),
              TCAST(rot_offset_for_block),
              TCAST(pose_for_rot),
              TCAST(block_type_ind_for_rot),
              TCAST(block_ind_for_rot),
              TCAST(sparse_inds),
              TCAST(sparse_energies));
        chunk_pair_offset_for_block_pair = std::get<0>(result).tensor;
        chunk_pair_offset = std::get<1>(result).tensor;
        energy2b = std::get<2>(result).tensor;
      }));

  std::vector<torch::Tensor> result(
      {chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b});
  return result;
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
    Tensor chunk_offset_offsets,
    Tensor chunk_offsets,
    Tensor energy1b,
    Tensor energy2b) {
  nvtx_range_push("pack_anneal");
  at::Tensor scores;
  at::Tensor rotamer_assignments;

  TMOL_DISPATCH_FLOATING_DEVICE(energy1b.options(), "pack_anneal", ([&] {
                                  constexpr tmol::Device Dev = device_t;

                                  std::cout << "HOLA!" << std::endl;
                                  auto result = AnnealerDispatch<Dev>::forward(
                                      max_n_rotamers_per_pose,
                                      TCAST(pose_n_res),
                                      TCAST(pose_n_rotamers),
                                      TCAST(pose_rotamer_offset),
                                      TCAST(n_rotamers_for_res),
                                      TCAST(oneb_offsets),
                                      TCAST(res_for_rot),
                                      chunk_size,
                                      TCAST(chunk_offset_offsets),
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
    TView<int64_t, 3, tmol::Device::CPU> chunk_offset_offsets,
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
          chunk_offset_offsets[pose],
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
    Tensor chunk_offset_offsets,
    Tensor chunk_offsets,
    Tensor energy1b,
    Tensor energy2b,
    Tensor rotamer_assignments) {
  auto result = compute_energies_for_assignments(
      TCAST(nrotamers_for_res),
      TCAST(oneb_offsets),
      int32_t(chunk_size),
      TCAST(chunk_offset_offsets),
      TCAST(chunk_offsets),
      TCAST(energy1b),
      TCAST(energy2b),
      TCAST(rotamer_assignments));
  return result.tensor;
}

/*
static auto registry = torch::jit::RegisterOperators()
                           .op("tmol::pack_anneal", &anneal)
                           .op("tmol::validate_energies", &validate_energies);
*/
// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("pack_anneal", &anneal);
  m.def("validate_energies", &validate_energies);
  m.def("build_interaction_graph", &build_interaction_graph);
}

}  // namespace compiled
}  // namespace pack
}  // namespace tmol
