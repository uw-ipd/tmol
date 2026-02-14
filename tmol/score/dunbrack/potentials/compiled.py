import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_dunbrack
dunbrack_pose_scores = _ops.dunbrack_pose_scores
dunbrack_rotamer_scores = _ops.dunbrack_rotamer_scores
