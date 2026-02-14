import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_cartbonded
cartbonded_pose_scores = _ops.cartbonded_pose_scores
cartbonded_rotamer_scores = _ops.cartbonded_rotamer_scores
