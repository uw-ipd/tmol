#pragma once

#include <torch/torch.h>

namespace tmol {
namespace score {
namespace common {

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
