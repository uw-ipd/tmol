import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_ljlk
ljlk_pose_scores = _ops.ljlk_pose_scores
ljlk_rotamer_scores = _ops.ljlk_rotamer_scores
