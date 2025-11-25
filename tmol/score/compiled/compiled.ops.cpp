#include <torch/torch.h>
#include <torch/script.h>

#include <tmol/utility/tensor/TensorCast.h>
#include <tmol/utility/function_dispatch/aten.hh>

#include <tmol/score/common/simple_dispatch.hh>
#include <tmol/score/common/forall_dispatch.hh>
#include <tmol/score/common/device_operations.hh>
#include <tmol/score/common/pose_score_fusion_module.hh>

namespace tmol {
namespace score {
namespace compiled {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class FusedScoreFunction
    : public torch::autograd::Function<FusedScoreFunction> {
 public:
  static Tensor forward(
      AutogradContext* ctx,
      // common params
      Tensor rot_coords,
      Tensor fused_sfxn_modules) {
    at::Tensor score;

    common::PoseScoreFusionModule* module0 =
        reinterpret_cast<common::PoseScoreFusionModule*>(
            fused_sfxn_modules[0].item<int64_t>());

    int const n_poses = module0->n_poses();
    int const max_n_blocks = module0->max_n_blocks();
    // int const n_atoms = rot_coords.size(0);

    int n_terms = 0;
    for (int i = 0; i < fused_sfxn_modules.size(0); i++) {
      common::PoseScoreFusionModule* module =
          reinterpret_cast<common::PoseScoreFusionModule*>(
              fused_sfxn_modules[i].item<int64_t>());
      n_terms += module->n_terms();
    }
    if (module0->output_block_pair_energies()) {
      score =
          rot_coords.new_zeros({n_terms, n_poses, max_n_blocks, max_n_blocks});
    } else {
      score = rot_coords.new_zeros({n_terms, n_poses, 1, 1});
    }
    // dscore_dcoords = rot_coords.new_zeros({n_terms, n_atoms, 3});

    // First pass: prepare for scoring
    // Figure out how much work there is to perform
    // Second pass may wait on events created in the first pass
    // TO DO: Setup a CUDA stream per module
    for (int i = 0; i < fused_sfxn_modules.size(0); i++) {
      common::PoseScoreFusionModule* module =
          reinterpret_cast<common::PoseScoreFusionModule*>(
              fused_sfxn_modules[i].item<int64_t>());
      module->prepare_for_scoring(rot_coords);
    }

    // Second pass: submit the kernels that will do the actual scoring
    // TO DO: Use the same CUDA streams oer module as in the setup pass
    int count_terms = 0;
    for (int i = 0; i < fused_sfxn_modules.size(0); i++) {
      common::PoseScoreFusionModule* module =
          reinterpret_cast<common::PoseScoreFusionModule*>(
              fused_sfxn_modules[i].item<int64_t>());
      int module_n_terms = module->n_terms();

      Tensor module_score = score.index(
          {torch::indexing::Slice(count_terms, count_terms + module_n_terms),
           torch::indexing::Slice(),
           torch::indexing::Slice(),
           torch::indexing::Slice()});
      module->forward(rot_coords, module_score);
      count_terms += module_n_terms;
    }
    ctx->save_for_backward({rot_coords, fused_sfxn_modules});

    // Get rid of last two singleton dimensions
    if (!module0->output_block_pair_energies()) {
      score = score.squeeze(-1).squeeze(-1);
    }

    return score;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();

    at::Tensor dV_d_pose_coords;

    at::Tensor rot_coords = saved[0];
    at::Tensor fused_sfxn_modules = saved[1];

    common::PoseScoreFusionModule* module0 =
        reinterpret_cast<common::PoseScoreFusionModule*>(
            fused_sfxn_modules[0].item<int64_t>());

    int const n_poses = module0->n_poses();
    int const max_n_blocks = module0->max_n_blocks();
    // int const n_atoms = rot_coords.size(0);

    int n_terms = 0;
    for (int i = 0; i < fused_sfxn_modules.size(0); i++) {
      common::PoseScoreFusionModule* module =
          reinterpret_cast<common::PoseScoreFusionModule*>(
              fused_sfxn_modules[i].item<int64_t>());
      n_terms += module->n_terms();
    }
    dV_d_pose_coords = rot_coords.new_zeros({rot_coords.size(0), 3});

    auto dTdV = grad_outputs[0];

    int count_terms = 0;
    for (int i = 0; i < fused_sfxn_modules.size(0); i++) {
      common::PoseScoreFusionModule* module =
          reinterpret_cast<common::PoseScoreFusionModule*>(
              fused_sfxn_modules[i].item<int64_t>());
      int module_n_terms = module->n_terms();

      Tensor module_dTdV;
      if (!module0->output_block_pair_energies()) {
        module_dTdV = dTdV.index({
            torch::indexing::Slice(count_terms, count_terms + module_n_terms),
            torch::indexing::Slice(),
        });
      } else {
        module_dTdV = dTdV.index(
            {torch::indexing::Slice(count_terms, count_terms + module_n_terms),
             torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice()});
      }
      // Tensor module_dV_d_pose_coords = dV_d_pose_coords.index(
      //     {torch::indexing::Slice(count_terms, count_terms + module_n_terms),
      //      torch::indexing::Slice()});

      module->backward(rot_coords, module_dTdV, dV_d_pose_coords);
      count_terms += module_n_terms;
    }

    return {dV_d_pose_coords, torch::Tensor()};
  }
};

Tensor fused_score_function(
    // common params
    Tensor rot_coords,
    Tensor fused_sfxn_modules) {
  return FusedScoreFunction::apply(rot_coords, fused_sfxn_modules);
}

void free_scoring_modules(Tensor fused_sfxn_modules) {
  for (int i = 0; i < fused_sfxn_modules.size(0); i++) {
    common::PoseScoreFusionModule* module =
        reinterpret_cast<common::PoseScoreFusionModule*>(
            fused_sfxn_modules[i].item<int64_t>());
    delete module;
    fused_sfxn_modules[i] = 0;
  }
}

// Macro indirection to force TORCH_EXTENSION_NAME macro expansion
// See https://stackoverflow.com/a/3221914
#define TORCH_LIBRARY_(ns, m) TORCH_LIBRARY(ns, m)
TORCH_LIBRARY_(TORCH_EXTENSION_NAME, m) {
  m.def("fused_score_function", &fused_score_function);
  m.def("free_scoring_modules", &free_scoring_modules);
}

}  // namespace compiled
}  // namespace score
}  // namespace tmol
