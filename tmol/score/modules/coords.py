import torch
from functools import singledispatch
from tmol.score.modules.bases import ScoreSystem


@singledispatch
def coords_for(val, score_system: ScoreSystem, *, requires_grad=True) -> torch.Tensor:
    """Shim function to extract forward-pass coords."""
    raise NotImplementedError(f"coords_for: {val}")
