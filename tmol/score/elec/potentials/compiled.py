import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_elec
elec_pose_scores = _ops.elec_pose_scores
elec_rotamer_scores = _ops.elec_rotamer_scores
