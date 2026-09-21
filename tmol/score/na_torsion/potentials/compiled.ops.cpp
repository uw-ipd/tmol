#include <torch/library.h>

#ifdef WITH_CUDA
#include <torch/torch.h>

#include "na_torsion_pose_score.hh"
#endif

namespace tmol {
namespace score {
namespace na_torsion {
namespace potentials {

#ifdef WITH_CUDA
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

class NaTorsionPoseScoreOp
    : public torch::autograd::Function<NaTorsionPoseScoreOp> {
 public:
  static std::vector<Tensor> forward(
      AutogradContext* ctx,
      Tensor coords,
      Tensor base,
      Tensor is_na,
      Tensor torsion_indices,
      Tensor torsion_ok,
      Tensor ring_indices,
      Tensor ring_ok,
      Tensor prev,
      Tensor backbone_means,
      Tensor backbone_sdev,
      Tensor sugar_means,
      Tensor chi_means,
      Tensor sdev_sugar,
      Tensor sdev_chi,
      Tensor well_pucker,
      Tensor well_alpha_gamma,
      Tensor well_bibii_pucker,
      Tensor well_alphanext_bibii,
      Tensor well_chi_syn,
      Tensor is_north,
      Tensor weight_bb,
      Tensor weight_chi,
      Tensor weight_sugar,
      double pucker_temperature,
      double bin_blend_sdev) {
    auto result = na_torsion_pose_score_cuda(
        coords,
        base,
        is_na,
        torsion_indices,
        torsion_ok,
        ring_indices,
        ring_ok,
        prev,
        backbone_means,
        backbone_sdev,
        sugar_means,
        chi_means,
        sdev_sugar,
        sdev_chi,
        well_pucker,
        well_alpha_gamma,
        well_bibii_pucker,
        well_alphanext_bibii,
        well_chi_syn,
        is_north,
        weight_bb,
        weight_chi,
        weight_sugar,
        pucker_temperature,
        bin_blend_sdev,
        coords.requires_grad());
    auto score = std::get<0>(result);
    auto derivatives = std::get<1>(result);
    ctx->save_for_backward({derivatives});
    ctx->mark_non_differentiable({derivatives});
    return {score, derivatives};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto grad_coords =
        na_torsion_pose_score_backward_cuda(saved[0], grad_outputs[0]);
    return {
        grad_coords, Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
        Tensor(),    Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
        Tensor(),    Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(),
        Tensor(),    Tensor(), Tensor(), Tensor(),
    };
  }
};

std::tuple<Tensor, Tensor> na_torsion_pose_score(
    Tensor coords,
    Tensor base,
    Tensor is_na,
    Tensor torsion_indices,
    Tensor torsion_ok,
    Tensor ring_indices,
    Tensor ring_ok,
    Tensor prev,
    Tensor backbone_means,
    Tensor backbone_sdev,
    Tensor sugar_means,
    Tensor chi_means,
    Tensor sdev_sugar,
    Tensor sdev_chi,
    Tensor well_pucker,
    Tensor well_alpha_gamma,
    Tensor well_bibii_pucker,
    Tensor well_alphanext_bibii,
    Tensor well_chi_syn,
    Tensor is_north,
    Tensor weight_bb,
    Tensor weight_chi,
    Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev) {
  auto result = NaTorsionPoseScoreOp::apply(
      coords,
      base,
      is_na,
      torsion_indices,
      torsion_ok,
      ring_indices,
      ring_ok,
      prev,
      backbone_means,
      backbone_sdev,
      sugar_means,
      chi_means,
      sdev_sugar,
      sdev_chi,
      well_pucker,
      well_alpha_gamma,
      well_bibii_pucker,
      well_alphanext_bibii,
      well_chi_syn,
      is_north,
      weight_bb,
      weight_chi,
      weight_sugar,
      pucker_temperature,
      bin_blend_sdev);
  return {result[0], result[1]};
}
#endif

TORCH_LIBRARY(tmol_na_torsion, m) {
#ifdef WITH_CUDA
  m.def("na_torsion_pose_score", &na_torsion_pose_score);
#else
  m.def(
      "na_torsion_pose_score("
      "Tensor coords, Tensor base, Tensor is_na, Tensor torsion_indices, "
      "Tensor torsion_ok, Tensor ring_indices, Tensor ring_ok, Tensor prev, "
      "Tensor backbone_means, Tensor backbone_sdev, Tensor sugar_means, "
      "Tensor chi_means, Tensor sdev_sugar, Tensor sdev_chi, "
      "Tensor well_pucker, Tensor well_alpha_gamma, "
      "Tensor well_bibii_pucker, Tensor well_alphanext_bibii, "
      "Tensor well_chi_syn, Tensor is_north, Tensor weight_bb, "
      "Tensor weight_chi, Tensor weight_sugar, float pucker_temperature, "
      "float bin_blend_sdev) -> (Tensor, Tensor)");
#endif
}

}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
