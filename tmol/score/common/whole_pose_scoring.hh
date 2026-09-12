#pragma once

#include <torch/torch.h>

namespace tmol {
namespace score {
namespace common {

#ifdef WITH_CUDA
// Defined in a separate CUDA translation unit so callers remain ordinary C++.
torch::Tensor accumulate_whole_pose_gradients_cuda(
    torch::Tensor const& saved_grad, torch::Tensor const& score_grad);
#endif

/// Apply per-score-type, per-pose upstream gradients to saved coordinate
/// derivatives from a whole-pose score kernel.
///
/// PoseStack coordinates are flattened from a dense [pose, atom, xyz] tensor,
/// so atoms belonging to each pose are contiguous. Reshaping to recover that
/// layout avoids the index_select kernel that the older implementation used
/// to expand each pose gradient to its atoms.
inline torch::Tensor accumulate_whole_pose_gradients(
    torch::Tensor const& saved_grad, torch::Tensor const& score_grad) {
  TORCH_INTERNAL_ASSERT(saved_grad.dim() == 3);
  TORCH_INTERNAL_ASSERT(score_grad.dim() == 2);
  TORCH_INTERNAL_ASSERT(saved_grad.size(0) == score_grad.size(0));

  int64_t const n_score_types = saved_grad.size(0);
  int64_t const n_poses = score_grad.size(1);
  int64_t const n_atoms = saved_grad.size(1);
  TORCH_INTERNAL_ASSERT(n_poses > 0 && n_atoms % n_poses == 0);

#ifdef WITH_CUDA
  // Keep Torch operations when constructing a derivative graph. The fused
  // kernel otherwise avoids materializing one weighted tensor per score type.
  // A scalar output uses a different Torch reduction order; lazy negative
  // views also require Torch to resolve their logical values.
  if (saved_grad.is_cuda() && !torch::GradMode::is_enabled()
      && saved_grad.is_contiguous() && n_score_types >= 2 && n_score_types <= 5
      && saved_grad.numel() > n_score_types && !saved_grad.is_neg()
      && !score_grad.is_neg() && saved_grad.device() == score_grad.device()
      && saved_grad.scalar_type() == score_grad.scalar_type()
      && (saved_grad.scalar_type() == torch::kFloat32
          || saved_grad.scalar_type() == torch::kFloat64)) {
    return accumulate_whole_pose_gradients_cuda(saved_grad, score_grad);
  }
#endif

  auto derivatives_by_pose = saved_grad.reshape(
      {n_score_types, n_poses, n_atoms / n_poses, saved_grad.size(2)});
  auto weighted =
      derivatives_by_pose * score_grad.reshape({n_score_types, n_poses, 1, 1});

  // Avoid launching a reduction for the many single-channel terms.
  auto accumulated =
      n_score_types == 1 ? weighted.select(0, 0) : weighted.sum(0);
  return accumulated.reshape({n_atoms, saved_grad.size(2)});
}

}  // namespace common
}  // namespace score
}  // namespace tmol
