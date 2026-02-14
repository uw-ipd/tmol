import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_constraint
get_torsion_angle = _ops.get_torsion_angle
