import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_disulfide
disulfide_pose_scores = _ops.disulfide_pose_scores
disulfide_rotamer_scores = _ops.disulfide_rotamer_scores
