#pragma once

#include <torch/torch.h>

namespace tmol {
namespace score {
namespace na_torsion {
namespace potentials {

std::tuple<at::Tensor, at::Tensor> na_torsion_pose_score_cuda(
    at::Tensor coords,
    at::Tensor base,
    at::Tensor is_na,
    at::Tensor torsion_indices,
    at::Tensor torsion_ok,
    at::Tensor ring_indices,
    at::Tensor ring_ok,
    at::Tensor prev,
    at::Tensor backbone_means,
    at::Tensor backbone_sdev,
    at::Tensor sugar_means,
    at::Tensor chi_means,
    at::Tensor sdev_sugar,
    at::Tensor sdev_chi,
    at::Tensor well_pucker,
    at::Tensor well_alpha_gamma,
    at::Tensor well_bibii_pucker,
    at::Tensor well_alphanext_bibii,
    at::Tensor well_chi_syn,
    at::Tensor is_north,
    at::Tensor weight_bb,
    at::Tensor weight_chi,
    at::Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev,
    bool compute_derivs);

at::Tensor na_torsion_pose_score_backward_cuda(
    at::Tensor derivatives, at::Tensor grad_output);

}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
