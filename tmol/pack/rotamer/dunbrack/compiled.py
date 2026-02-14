import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_dun_sampler
dun_sample_chi = _ops.dun_sample_chi
