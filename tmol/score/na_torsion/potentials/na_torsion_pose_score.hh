#include <tmol/utility/tensor/native.hh>
#pragma once

namespace tmol {
namespace score {
namespace na_torsion {
namespace potentials {

std::tuple<tmol::native::Tensor, tmol::native::Tensor>
na_torsion_pose_score_cuda(
    tmol::native::Tensor coords,
    tmol::native::Tensor base,
    tmol::native::Tensor is_na,
    tmol::native::Tensor torsion_indices,
    tmol::native::Tensor torsion_ok,
    tmol::native::Tensor ring_indices,
    tmol::native::Tensor ring_ok,
    tmol::native::Tensor prev,
    tmol::native::Tensor backbone_means,
    tmol::native::Tensor backbone_sdev,
    tmol::native::Tensor sugar_means,
    tmol::native::Tensor chi_means,
    tmol::native::Tensor sdev_sugar,
    tmol::native::Tensor sdev_chi,
    tmol::native::Tensor well_pucker,
    tmol::native::Tensor well_alpha_gamma,
    tmol::native::Tensor well_bibii_pucker,
    tmol::native::Tensor well_alphanext_bibii,
    tmol::native::Tensor well_chi_syn,
    tmol::native::Tensor is_north,
    tmol::native::Tensor weight_bb,
    tmol::native::Tensor weight_chi,
    tmol::native::Tensor weight_sugar,
    double pucker_temperature,
    double bin_blend_sdev,
    bool compute_derivs);

tmol::native::Tensor na_torsion_pose_score_backward_cuda(
    tmol::native::Tensor derivatives, tmol::native::Tensor grad_output);

}  // namespace potentials
}  // namespace na_torsion
}  // namespace score
}  // namespace tmol
