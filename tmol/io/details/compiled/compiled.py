import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_io
gen_pose_leaf_atoms = _ops.gen_pose_leaf_atoms
resolve_his_taut = _ops.resolve_his_taut
