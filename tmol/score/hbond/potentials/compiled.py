import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_hbond
hbond_pose_scores = _ops.hbond_pose_scores
hbond_rotamer_scores = _ops.hbond_rotamer_scores
