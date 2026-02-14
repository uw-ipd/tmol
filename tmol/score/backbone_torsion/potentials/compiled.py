import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_bb_torsion
backbone_torsion_pose_score = _ops.backbone_torsion_pose_score
backbone_torsion_rotamer_score = _ops.backbone_torsion_rotamer_score
